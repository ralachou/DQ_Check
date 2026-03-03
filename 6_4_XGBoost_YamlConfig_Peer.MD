Here’s a second YAML file you can plug next to your dq_model_catalog.yaml, dedicated to peer-group definitions (sector / rating / tenor / curve / surface peers) by asset class, with clear join keys, fallback rules, and minimum peer sizes.

I’m using industry taxonomies like GICS (4-level hierarchy) and NAICS (U.S. government standard) as optional peer dimensions when you have issuer metadata.
For credit tenors, the YAML supports standard CDS buckets often used in practice (e.g., 3Y/5Y/7Y/10Y, and where available 6M/1Y/3Y/5Y/7Y/10Y).

# dq_peer_groups.yaml
peer_catalog_version: "1.0"
owner: "RFDM-DQ-ControlTower"
last_updated_utc: "2026-03-03T00:00:00Z"

global:
  rf_path_fields: ["rf_level1","rf_level2","rf_level3","rf_level4","rf_level5"]
  default_min_peer_count: 25
  default_max_peer_count: 5000
  # If peer group too small, progressively relax dimensions in this order
  default_fallback_chain:
    - "strict"
    - "drop_one_dim"
    - "drop_two_dims"
    - "assetclass_only"
  aggregation:
    center: "median"
    dispersion: "mad"   # median absolute deviation (robust)
  matching:
    # Used when tenors/strikes are slightly off (e.g., 59d vs 60d)
    numeric_tolerance:
      tenor_days: 7
      strike_pct: 0.25
      delta: 0.02
  excludes:
    # exclude the series itself from peer sets by rf_id
    self_exclude: true

# -------------------------
# Metadata inputs (enrichment tables)
# -------------------------
enrichment_sources:
  issuer_master:
    keys: ["issuer_id"]
    fields:
      - "sector_gics"         # optional; GICS sector/industry/subindustry supported :contentReference[oaicite:2]{index=2}
      - "industry_gics"
      - "subindustry_gics"
      - "naics"               # optional; NAICS hierarchy supported :contentReference[oaicite:3]{index=3}
      - "country"
      - "region"
  credit_instrument_master:
    keys: ["instrument_id"]
    fields:
      - "issuer_id"
      - "rating_bucket"       # internal mapping AAA..CCC, NR
      - "seniority"           # SR, SUB, etc.
      - "currency"
      - "doc_clause"          # ISDA/contractual optional
      - "liquidity_bucket"    # e.g., TRACE/volume-based internal
  equity_master:
    keys: ["ticker"]
    fields:
      - "issuer_id"
      - "exchange"
      - "country"
      - "market_cap_bucket"
  rates_master:
    keys: ["curve_id"]
    fields:
      - "currency"
      - "curve_family"        # OIS, IBOR, GOVVY, SWAP, etc.
      - "provider"
  fx_master:
    keys: ["ccy_pair"]
    fields:
      - "base_ccy"
      - "quote_ccy"
      - "region_pair"
  vol_surface_master:
    keys: ["surface_id"]
    fields:
      - "underlying_id"
      - "surface_type"        # smile / skew / term-structure
      - "quote_convention"    # delta/strike/ATM/RR/BF

# -------------------------
# Peer rules by asset class (rf_level1)
# -------------------------
asset_classes:

  # =========================
  # CREDIT SPREADS / CORP CURVES (CP_RATE)
  # =========================
  CP_RATE:
    description: "Credit spreads/curves for issuers, sectors, ratings, seniority, tenor"
    peer_sets:

      - peer_set_id: "cp_issuer_tenor_strict"
        purpose: "Issuer-level peer for detecting bad points vs same issuer curve"
        join:
          dims:
            - {name: "issuer_id", source: "credit_instrument_master"}
            - {name: "currency",  source: "credit_instrument_master"}
            - {name: "seniority", source: "credit_instrument_master"}
            - {name: "tenor_bucket", source: "rf_level5", type: "tenor_bucket"}
        constraints:
          min_peer_count: 8      # issuer curve can be smaller
          max_peer_count: 200
        fallback_chain:
          - rule: "drop_dim"
            dim: "seniority"
          - rule: "drop_dim"
            dim: "currency"
          - rule: "switch_to_peer_set"
            peer_set_id: "cp_sector_rating_tenor"

      - peer_set_id: "cp_sector_rating_tenor"
        purpose: "Cross-issuer peer: sector x rating x tenor"
        join:
          dims:
            - {name: "sector_gics", source: "issuer_master"}   # optional taxonomy :contentReference[oaicite:4]{index=4}
            - {name: "rating_bucket", source: "credit_instrument_master"}
            - {name: "currency", source: "credit_instrument_master"}
            - {name: "tenor_bucket", source: "rf_level5", type: "tenor_bucket"}
        constraints:
          min_peer_count: 25
          max_peer_count: 3000
        fallback_chain:
          - rule: "drop_dim"
            dim: "sector_gics"
          - rule: "drop_dim"
            dim: "currency"
          - rule: "drop_dim"
            dim: "rating_bucket"

      - peer_set_id: "cp_index_curve_peer"
        purpose: "Benchmark issuer vs index curve (CDX/iTraxx-like) by rating/region"
        join:
          dims:
            - {name: "region", source: "issuer_master"}
            - {name: "rating_bucket", source: "credit_instrument_master"}
            - {name: "tenor_bucket", source: "rf_level5", type: "tenor_bucket"}
        constraints:
          min_peer_count: 50
          max_peer_count: 10000

    tenor_bucketing:
      # Supports common CDS curve tenors used in practice (you can extend)
      # e.g., 3Y/5Y/7Y/10Y are standard for many index tranches; 6M/1Y also used for single-name curves :contentReference[oaicite:5]{index=5}
      method: "nearest_standard"
      standard_tenors:
        - {label: "6M", days: 182}
        - {label: "1Y", days: 365}
        - {label: "3Y", days: 1095}
        - {label: "5Y", days: 1825}
        - {label: "7Y", days: 2555}
        - {label: "10Y", days: 3650}

  # =========================
  # INTEREST RATES (IR_RATE)
  # =========================
  IR_RATE:
    description: "OIS/IBOR/Govvie/swap curves by currency and curve family"
    peer_sets:

      - peer_set_id: "ir_curve_family_currency_node"
        purpose: "Same curve family (OIS vs SWAP vs GOVVY) within currency; compare node behavior"
        join:
          dims:
            - {name: "currency", source: "rates_master"}
            - {name: "curve_family", source: "rates_master"}
            - {name: "tenor_bucket", source: "rf_level5", type: "tenor_bucket"}
        constraints:
          min_peer_count: 15
          max_peer_count: 5000
        fallback_chain:
          - rule: "drop_dim"
            dim: "curve_family"

      - peer_set_id: "ir_cross_provider_consistency"
        purpose: "Same currency/family/tenor across providers (vendor check)"
        join:
          dims:
            - {name: "currency", source: "rates_master"}
            - {name: "curve_family", source: "rates_master"}
            - {name: "provider", source: "rates_master"}
            - {name: "tenor_bucket", source: "rf_level5", type: "tenor_bucket"}
        constraints:
          min_peer_count: 5
          max_peer_count: 200

  # =========================
  # FX SPOT (FX_RATE)
  # =========================
  FX_RATE:
    description: "FX spot by currency pair and optionally region bucket"
    peer_sets:

      - peer_set_id: "fx_pair_strict"
        purpose: "Same pair only (for staleness/missing/outlier detection)"
        join:
          dims:
            - {name: "ccy_pair", source: "fx_master"}
        constraints:
          min_peer_count: 1
          max_peer_count: 1

      - peer_set_id: "fx_region_pair_peer"
        purpose: "Compare pair vs region bucket (e.g., EM vs G10) and USD-legs"
        join:
          dims:
            - {name: "region_pair", source: "fx_master"}
            - {name: "has_usd_leg", source: "fx_master", derive: "base_ccy=='USD' OR quote_ccy=='USD'"}
        constraints:
          min_peer_count: 20
          max_peer_count: 5000

  # =========================
  # EQUITY PRICES (EQ_PRICE)
  # =========================
  EQ_PRICE:
    description: "Equity spot/returns with sector and market-cap peer groups"
    peer_sets:

      - peer_set_id: "eq_sector_mcap_country"
        purpose: "Equity peer by sector + market cap + country"
        join:
          dims:
            - {name: "sector_gics", source: "issuer_master"}    # taxonomy reference :contentReference[oaicite:6]{index=6}
            - {name: "market_cap_bucket", source: "equity_master"}
            - {name: "country", source: "issuer_master"}
        constraints:
          min_peer_count: 30
          max_peer_count: 5000
        fallback_chain:
          - rule: "drop_dim"
            dim: "country"
          - rule: "drop_dim"
            dim: "market_cap_bucket"
          - rule: "drop_dim"
            dim: "sector_gics"

  # =========================
  # VOLATILITY (EQ_VOL / FX_VOL / IR_VOL / COMM_VOL)
  # =========================
  VOL:
    description: "Generic vol surface peers; specialized by underlying + tenor + delta/strike"
    applies_to_rf_level1: ["EQ_VOL","FX_VOL","IR_VOL","COMM_VOL"]
    peer_sets:

      - peer_set_id: "vol_underlying_tenor_delta"
        purpose: "Same underlying & tenor; compare smile points by delta/strike neighborhood"
        join:
          dims:
            - {name: "underlying_id", source: "vol_surface_master"}
            - {name: "tenor_bucket", source: "rf_level5", type: "tenor_bucket"}
            - {name: "quote_convention", source: "vol_surface_master"}
            - {name: "delta_bucket", source: "rf_level4", type: "delta_bucket"} # e.g., 10D/25D/ATM
        constraints:
          min_peer_count: 10
          max_peer_count: 5000
        fallback_chain:
          - rule: "drop_dim"
            dim: "delta_bucket"
          - rule: "drop_dim"
            dim: "quote_convention"

    delta_bucketing:
      method: "nearest_standard"
      standards:
        - {label: "10D", value: 0.10}
        - {label: "25D", value: 0.25}
        - {label: "ATM", value: 0.50}
        - {label: "75D", value: 0.75}
        - {label: "90D", value: 0.90}

  # =========================
  # COMMODITIES (COMM_PRICE)
  # =========================
  COMM_PRICE:
    description: "Commodity forwards by commodity, location/grade, and contract month/tenor"
    peer_sets:

      - peer_set_id: "comm_commodity_location_tenor"
        purpose: "Same commodity family (e.g., WTI/Brent), location/grade, and tenor"
        join:
          dims:
            - {name: "commodity_family", source: "rf_level3"}
            - {name: "location_or_grade", source: "rf_level4"}
            - {name: "tenor_bucket", source: "rf_level5", type: "tenor_bucket"}
        constraints:
          min_peer_count: 15
          max_peer_count: 3000
        fallback_chain:
          - rule: "drop_dim"
            dim: "location_or_grade"

# -------------------------
# Peer quality filters (optional but recommended)
# -------------------------
peer_quality_filters:
  liquidity:
    enabled: true
    method: "internal_bucket"
    min_liquidity_bucket: "L2"
  data_completeness:
    enabled: true
    min_non_missing_frac_lookback: 0.9
  exclude_known_bad_sources:
    enabled: true
    fields:
      - "vendor_id"
      - "provider"

# -------------------------
# Output contract for peer computation
# -------------------------
outputs:
  peer_membership_table: "dq_peer_membership"   # (peer_set_id, asof_date, rf_id, peer_rf_id)
  peer_stats_table: "dq_peer_stats"             # (peer_set_id, asof_date, rf_id, peer_median, peer_mad, peer_percentile, n_peers)

How you’d actually use this in the scoring layer

For each model_id that declares feature_set: fs_peer_context_v1, your feature builder:

picks the correct peer_set_id for that rf_level1

materializes peer members (with fallback if too small)

computes peer stats (median/MAD/percentile)

emits features like peer_robust_z, peer_dev, peer_percentile

This “peer rulebook” is the missing link that makes sector/rating/tenor comparisons systematic and auditable.