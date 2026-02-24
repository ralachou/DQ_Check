from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from data_access.parquet_repository import ParquetTimeSeriesRepository
from engine.run_engine import RunEngine, RunRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_CANONICAL = [
    "business_date",
    "risk_factor_id",
    "close_date",
    "close_value",
    "rf_level1",
    "rf_level2",
    "rf_level3",
    "rf_level4",
    "rf_level5",
]

COLUMN_CANDIDATES: dict[str, list[str]] = {
    "business_date": ["business dates", "businessDate", "business_date"],
    "risk_factor_id": ["driverName", "risk_factor_id"],
    "close_date": ["closeDate", "close_date", "date"],
    "close_value": ["closeValue", "close_value", "value"],
    "prev_close_date": ["prevCLoseDate", "prevCloseDate", "prev_close_date"],
    "prev_close_value": ["preCloseValue", "prevCloseValue", "prev_close_value"],
    "shift_src": ["shifts", "shift", "return"],
    "asset_class": ["assetClass", "asset_class"],
    "ccy": ["ccy", "currency"],
    "rf_level1": ["rf_level1"],
    "rf_level2": ["rf_level2"],
    "rf_level3": ["rf_level3"],
    "rf_level4": ["rf_level4"],
    "rf_level5": ["rf_level5"],
}


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    by_lower = {str(c).strip().lower(): str(c) for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in by_lower:
            return by_lower[key]
    return None


def _resolve_columns(df: pd.DataFrame) -> dict[str, str | None]:
    resolved: dict[str, str | None] = {}
    for canonical, candidates in COLUMN_CANDIDATES.items():
        resolved[canonical] = _find_column(df, candidates)
    return resolved


def _load_source(path: Path, sheet: str | int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input extension '{suffix}'. Use .csv/.xlsx/.xls.")


def _build_canonical(df: pd.DataFrame, cols: dict[str, str | None]) -> pd.DataFrame:
    out = pd.DataFrame()
    for key in REQUIRED_CANONICAL + ["prev_close_date", "prev_close_value", "shift_src", "asset_class", "ccy"]:
        src_col = cols.get(key)
        out[key] = df[src_col] if src_col else pd.NA

    out["business_date"] = pd.to_datetime(out["business_date"], errors="coerce").dt.normalize()
    out["risk_factor_id"] = out["risk_factor_id"].astype("string")
    out["close_date"] = pd.to_datetime(out["close_date"], errors="coerce")
    out["close_value"] = pd.to_numeric(out["close_value"], errors="coerce")
    out["prev_close_date"] = pd.to_datetime(out["prev_close_date"], errors="coerce")
    out["prev_close_value"] = pd.to_numeric(out["prev_close_value"], errors="coerce")
    out["shift_src"] = pd.to_numeric(out["shift_src"], errors="coerce")

    for c in ["rf_level1", "rf_level2", "rf_level3", "rf_level4", "rf_level5", "asset_class", "ccy"]:
        out[c] = out[c].astype("string")
    return out


def _build_timeseries(canonical: pd.DataFrame) -> pd.DataFrame:
    cur = canonical[["business_date", "risk_factor_id", "close_date", "close_value"]].rename(
        columns={"close_date": "date", "close_value": "value"}
    )
    prev = canonical[["business_date", "risk_factor_id", "prev_close_date", "prev_close_value"]].rename(
        columns={"prev_close_date": "date", "prev_close_value": "value"}
    )
    ts = pd.concat([cur, prev], ignore_index=True)
    ts = ts.dropna(subset=["business_date", "risk_factor_id", "date", "value"]).copy()
    ts = ts.drop_duplicates(subset=["business_date", "risk_factor_id", "date"], keep="last")
    ts = ts.sort_values(["business_date", "risk_factor_id", "date"]).reset_index(drop=True)
    return ts


def _build_metadata(canonical: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "risk_factor_id",
        "asset_class",
        "ccy",
        "rf_level1",
        "rf_level2",
        "rf_level3",
        "rf_level4",
        "rf_level5",
    ]
    meta = canonical[cols].drop_duplicates(subset=["risk_factor_id"]).copy()
    meta["rf_level1"] = meta["rf_level1"].fillna(meta["asset_class"])
    meta["rf_level2"] = meta["rf_level2"].fillna(meta["ccy"])
    for c in ["rf_level1", "rf_level2", "rf_level3", "rf_level4", "rf_level5"]:
        meta[c] = meta[c].fillna("NA").astype(str).str.strip().replace("", "NA")
    meta["risk_factor_id"] = meta["risk_factor_id"].astype(str)
    return meta


def _reset_paths(raw_path: Path, processed_path: Path, artifact_root: Path) -> None:
    def _handle_remove_readonly(func, path, exc_info):
        del exc_info
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            pass

    targets = [
        raw_path / "timeseries_raw",
        raw_path / "risk_factors",
        processed_path / "dq_results",
        processed_path / "universe_membership",
        processed_path / "audit_log",
        artifact_root,
    ]
    for target in targets:
        if target.exists():
            shutil.rmtree(target, onerror=_handle_remove_readonly)


def _write_raw_tables(ts: pd.DataFrame, meta: pd.DataFrame, raw_path: Path) -> dict[str, Any]:
    ts_root = raw_path / "timeseries_raw"
    rf_root = raw_path / "risk_factors"
    ts_root.mkdir(parents=True, exist_ok=True)
    rf_root.mkdir(parents=True, exist_ok=True)

    ts_files: list[str] = []
    for bd, part in ts.groupby(ts["business_date"].dt.strftime("%Y-%m-%d"), sort=True):
        out = ts_root / f"business_date={bd}"
        out.mkdir(parents=True, exist_ok=True)
        file_path = out / "part-000.parquet"
        part.drop(columns=["business_date"]).to_parquet(file_path, index=False)
        ts_files.append(str(file_path))

    rf_path = rf_root / "risk_factors.parquet"
    meta.to_parquet(rf_path, index=False)
    return {"timeseries_files": ts_files, "risk_factors_file": str(rf_path)}


def _compatible_universes(meta: pd.DataFrame, config_path: Path) -> list[str]:
    if not config_path.exists():
        return []
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    universes = cfg.get("universes", {})
    rf1_vals = set(meta["rf_level1"].dropna().astype(str).unique().tolist())
    out: list[str] = []
    for name, ucfg in universes.items():
        filters = (ucfg or {}).get("filters", {})
        rf1 = filters.get("rf_level1")
        if rf1 is None or rf1 in rf1_vals:
            out.append(str(name))
    return sorted(out)


def _date_or_default(value: str, default: str) -> str:
    text = str(value or "").strip()
    if not text:
        return default
    return str(pd.to_datetime(text).date())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrate manual MAR-style market data into raw parquet tables and optionally run the DQ engine."
    )
    parser.add_argument(
        "--source-path",
        default=str(PROJECT_ROOT / "data" / "incoming" / "manual_dataset.xlsx"),
        help="Manual dataset path (.csv/.xlsx/.xls).",
    )
    parser.add_argument("--source-sheet", default="0", help="Sheet name/index for Excel input.")
    parser.add_argument("--raw-path", default=str(PROJECT_ROOT / "data" / "raw_manual"))
    parser.add_argument("--processed-path", default=str(PROJECT_ROOT / "data" / "processed_manual"))
    parser.add_argument("--artifact-root", default=str(PROJECT_ROOT / "data" / "artifacts" / "normalization"))
    parser.add_argument("--config-path", default=str(PROJECT_ROOT / "configs" / "model_catalog.yaml"))
    parser.add_argument("--universe-name", default="CP_RATE_CORP")
    parser.add_argument(
        "--universe-names",
        nargs="*",
        default=[],
        help="Optional explicit list of universes to run (overrides --universe-name).",
    )
    parser.add_argument(
        "--run-all-compatible-universes",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When --run-engine is set, run all compatible universes inferred from staged asset classes.",
    )
    parser.add_argument("--run-id", default="", help="Optional explicit run id for engine execution.")
    parser.add_argument("--start-date", default="", help="Optional override; default=min staged timeseries date.")
    parser.add_argument("--end-date", default="", help="Optional override; default=max staged timeseries date.")
    parser.add_argument(
        "--business-date",
        default="",
        help="Optional override; default=max staged business_date. Used for engine + UI launch command.",
    )
    parser.add_argument(
        "--run-engine",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run Layer 5 engine after staging raw tables.",
    )
    parser.add_argument(
        "--clean-target",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Delete previous raw/processed target folders before staging new data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_path)
    raw_path = Path(args.raw_path)
    processed_path = Path(args.processed_path)
    artifact_root = Path(args.artifact_root)
    config_path = Path(args.config_path)

    if args.clean_target:
        _reset_paths(raw_path=raw_path, processed_path=processed_path, artifact_root=artifact_root)

    sheet_arg: str | int
    if str(args.source_sheet).isdigit():
        sheet_arg = int(args.source_sheet)
    else:
        sheet_arg = str(args.source_sheet)

    source_df = _load_source(source_path, sheet=sheet_arg)
    columns = _resolve_columns(source_df)
    missing = [k for k in REQUIRED_CANONICAL if columns.get(k) is None]
    if missing:
        raise ValueError(f"Missing required source fields: {missing}. Resolved columns={columns}")

    canonical = _build_canonical(source_df, columns)
    timeseries = _build_timeseries(canonical)
    metadata = _build_metadata(canonical)
    if timeseries.empty:
        raise ValueError("Mapped timeseries is empty after cleaning. Check source mapping and date/value columns.")
    if metadata.empty:
        raise ValueError("Mapped risk_factors metadata is empty after cleaning.")

    written = _write_raw_tables(timeseries, metadata, raw_path=raw_path)
    ts_min = str(pd.to_datetime(timeseries["date"]).min().date())
    ts_max = str(pd.to_datetime(timeseries["date"]).max().date())
    biz_max = str(pd.to_datetime(timeseries["business_date"]).max().date())

    start_date = _date_or_default(args.start_date, ts_min)
    end_date = _date_or_default(args.end_date, ts_max)
    business_date = _date_or_default(args.business_date, biz_max)
    compatible_universes = _compatible_universes(metadata, config_path=config_path)

    run_summary: dict[str, Any] = {
        "layer": "Manual dataset integration demo (no framework code changes)",
        "source_path": str(source_path.resolve()),
        "raw_path": str(raw_path.resolve()),
        "processed_path": str(processed_path.resolve()),
        "input_columns": {k: v for k, v in columns.items()},
        "rows": {
            "source_rows": int(len(source_df)),
            "timeseries_rows": int(len(timeseries)),
            "risk_factor_rows": int(len(metadata)),
            "timeseries_unique_business_dates": int(timeseries["business_date"].nunique()),
            "timeseries_unique_risk_factors": int(timeseries["risk_factor_id"].nunique()),
        },
        "date_bounds": {"start_date": start_date, "end_date": end_date, "business_date": business_date},
        "files_written": written,
        "compatible_universes": compatible_universes[:20],
        "engine_run": {"executed": False},
    }

    if args.run_engine:
        explicit_universes = [str(u).strip() for u in args.universe_names if str(u).strip()]
        if args.run_all_compatible_universes:
            universe_list = compatible_universes
        elif explicit_universes:
            universe_list = explicit_universes
        else:
            universe_list = [str(args.universe_name).strip()]

        if not universe_list:
            raise ValueError(
                "No universes selected to run. Provide --universe-name/--universe-names "
                "or use --run-all-compatible-universes with compatible staged asset classes."
            )

        run_id = args.run_id.strip() or f"integrate_mar_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
        repo = ParquetTimeSeriesRepository(base_path=raw_path, write_base_path=processed_path)
        engine = RunEngine(repository=repo, artifact_root=str(artifact_root))

        successes: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        for universe_name in universe_list:
            try:
                engine_out = engine.run(
                    RunRequest(
                        run_id=run_id,
                        universe_name=universe_name,
                        start_date=start_date,
                        end_date=end_date,
                        business_date=business_date,
                        config_path=str(config_path),
                        incremental=False,
                    )
                )
                result_rows = int(len(engine_out["results"]))
                flagged_rows = int(engine_out["results"]["flag"].sum()) if "flag" in engine_out["results"].columns else 0
                successes.append(
                    {
                        "universe_name": universe_name,
                        "result_rows": result_rows,
                        "flagged_rows": flagged_rows,
                    }
                )
            except Exception as exc:
                failures.append({"universe_name": universe_name, "error": str(exc)})

        run_summary["engine_run"] = {
            "executed": True,
            "run_id": run_id,
            "mode": (
                "all_compatible"
                if args.run_all_compatible_universes
                else ("explicit_list" if explicit_universes else "single")
            ),
            "requested_universes": universe_list,
            "successful_universes": [x["universe_name"] for x in successes],
            "failed_universes": failures,
            "total_result_rows": int(sum(x["result_rows"] for x in successes)),
            "total_flagged_rows": int(sum(x["flagged_rows"] for x in successes)),
            "per_universe": successes,
        }

    out_root = processed_path / "integration_with_mar_demo" / f"business_date={business_date}"
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print("=== Integration With MAR Dataset ===")
    print(f"Source rows: {len(source_df):,}")
    print(f"Staged timeseries rows: {len(timeseries):,}")
    print(f"Staged risk_factors rows: {len(metadata):,}")
    print(f"Date range: {start_date} -> {end_date}")
    print(f"Business date: {business_date}")
    print(f"Raw staging path: {raw_path}")
    print(f"Processed path: {processed_path}")
    if compatible_universes:
        print(f"Compatible universes (first 10): {compatible_universes[:10]}")
    else:
        print("Compatible universes: none found in config.")
    print(f"Summary file: {summary_path}")
    print(
        "Launch UI command: "
        f"python -m ui.app_v2 --results-path {processed_path / 'dq_results'} "
        f"--raw-path {raw_path / 'timeseries_raw'} "
        f"--membership-path {processed_path / 'universe_membership'} "
        f"--config-path {config_path} --business-date {business_date}"
    )


if __name__ == "__main__":
    main()
