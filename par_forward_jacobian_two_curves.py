"""
================================================================================
PAR -> FORWARD JACOBIAN, two ways of building the curve.
   Method A : piecewise-constant instantaneous forward  (local bootstrap)
   Method B : cubic spline on zero rates                (global smoothing)
Same 12 OIS quotes, same "reprice to par" constraint for both. The ONLY
difference is how we interpolate between pillars -> two different Jacobians.

Workflow per column j:  bump par_j by 1bp -> rebuild curve -> reprice ->
recompute forward buckets -> delta = d(forward)/d(par_j).
================================================================================
"""
import numpy as np
from scipy.interpolate import CubicSpline
np.set_printoptions(precision=3, suppress=True, linewidth=200)

# ---- market ----------------------------------------------------------------
PLBL = ['1M','3M','6M','9M','1Y','2Y','3Y','4Y','5Y','10Y','20Y','30Y']   # par tenors
T    = np.array([1/12,0.25,0.5,0.75,1,2,3,4,5,10,20,30], float)
PAR  = np.array([4.30,4.25,4.15,4.05,3.95,3.80,3.75,3.78,3.82,4.00,4.15,4.10])/100
N    = len(T)
SLBL = ['0-1M','1M-3M','3M-6M','6M-9M','9M-1Y','1-2Y','2-3Y','3-4Y',       # fwd segments
        '4-5Y','5-10Y','10-20Y','20-30Y']

def schedule(i):
    Ti = T[i]
    if Ti < 1.0: return np.array([Ti]), np.array([Ti])
    n = int(round(Ti)); return np.arange(1,n+1,dtype=float), np.ones(n)

# ============================================================================
# METHOD A : piecewise-constant instantaneous forward, sequential bootstrap
# ============================================================================
def DF_pw(t, f):
    if t <= 0: return 1.0
    df, prev = 1.0, 0.0
    for k in range(N):
        end = T[k]
        if t >= end: df *= np.exp(-f[k]*(end-prev)); prev = end
        else: return df*np.exp(-f[k]*(t-prev))
    return df*np.exp(-f[-1]*(t-prev))

def bootstrap_pw(par):
    f = np.zeros(N)
    for k in range(N):
        lo, hi = -0.5, 1.0
        for _ in range(100):
            mid = .5*(lo+hi); f[k] = mid
            times, accr = schedule(k)
            val = par[k]*sum(a*DF_pw(t,f) for t,a in zip(times,accr)) - (1-DF_pw(T[k],f))
            if val > 0: lo = mid
            else: hi = mid
        f[k] = .5*(lo+hi)
    return lambda t, ff=f.copy(): DF_pw(t, ff)          # returns a DF(t) function

# ============================================================================
# METHOD B : cubic spline on zero rates, calibrated GLOBALLY to reprice par
# ============================================================================
def DF_sp(t, cs):
    return 1.0 if t <= 0 else np.exp(-float(cs(t))*t)

def swapval_sp(i, cs, par):
    times, accr = schedule(i)
    return par[i]*sum(a*DF_sp(t,cs) for t,a in zip(times,accr)) - (1-DF_sp(T[i],cs))

def calibrate_sp(par, z0):
    z = z0.copy()
    for _ in range(60):
        cs = CubicSpline(T, z, bc_type='natural')
        R  = np.array([swapval_sp(i, cs, par) for i in range(N)])
        if np.max(np.abs(R)) < 1e-15: break
        J, eps = np.zeros((N,N)), 1e-7
        for k in range(N):
            zk = z.copy(); zk[k] += eps
            csk = CubicSpline(T, zk, bc_type='natural')
            J[:,k] = (np.array([swapval_sp(i, csk, par) for i in range(N)]) - R)/eps
        z = z + np.linalg.solve(J, -R)
    cs = CubicSpline(T, z, bc_type='natural')
    return lambda t, c=cs: DF_sp(t, c), cs

# ============================================================================
# forward buckets (inter-pillar continuous forward) from ANY DF(t) function
# ============================================================================
def fwd_buckets(dffun):
    f = np.zeros(N); prev_t, prev_df = 0.0, 1.0
    for k in range(N):
        d = dffun(T[k])
        f[k] = (np.log(prev_df) - np.log(d))/(T[k]-prev_t)
        prev_t, prev_df = T[k], d
    return f

# ============================================================================
# Jacobian by bump-reprice-delta.  builder(par) -> DF(t) function.
# ============================================================================
def jacobian(builder, par, bp=1e-4):
    base = fwd_buckets(builder(par))
    J = np.zeros((N, N))
    for j in range(N):
        pj = par.copy(); pj[j] += bp
        J[:, j] = (fwd_buckets(builder(pj)) - base)/bp        # bp fwd per bp par
    return J, base

def build_pw(par):  return bootstrap_pw(par)
def build_sp(par):
    z0 = fwd_buckets(bootstrap_pw(par))          # warm start from pw zero-ish level
    z0 = -np.log([bootstrap_pw(par)(t) for t in T])/T
    return calibrate_sp(par, z0)[0]

# ============================================================================
# RUN
# ============================================================================
Ja, base_a = jacobian(build_pw, PAR)
Jb, base_b = jacobian(build_sp, PAR)

def show_pattern(J, name):
    print(f"\n{name}: nonzero pattern (X = |dF/dpar| > 0.01 bp/bp)")
    print("            " + " ".join(f"{p:>4}" for p in PLBL))
    for k in range(N):
        row = " ".join("   X" if abs(J[k,j])>0.01 else "   ." for j in range(N))
        print(f"{SLBL[k]:>10}  {row}")

print("="*90)
print("PAR -> FORWARD JACOBIAN  (rows = forward segment, cols = par tenor, bp/bp)")
print("="*90)
show_pattern(Ja, "METHOD A  piecewise-constant forward")
show_pattern(Jb, "METHOD B  cubic spline on zeros")

# locality metric: how many par tenors does each forward bucket depend on?
def support(J, thr=0.01):
    return (np.abs(J) > thr).sum(axis=1)          # nonzeros per row (per fwd bucket)
sa, sb = support(Ja), support(Jb)
print("\nLOCALITY  —  how many par tenors each forward bucket depends on:")
print(f"   Method A (piecewise): avg {sa.mean():.1f} par tenors per bucket  (max {sa.max()})")
print(f"   Method B (spline)   : avg {sb.mean():.1f} par tenors per bucket  (max {sb.max()})")
row = SLBL.index('10-20Y')
depA = [PLBL[j] for j in range(N) if abs(Ja[row,j])>0.01]
depB = [PLBL[j] for j in range(N) if abs(Jb[row,j])>0.01]
print(f"\n   The 10-20Y forward bucket depends on:")
print(f"      Method A: {depA}")
print(f"      Method B: {depB}   <- spline reaches all the way back to the front end")

# focused column: what does a +1bp 10Y par bump do to every forward bucket?
c = PLBL.index('10Y')
print("\n" + "="*90)
print("FOCUS: +1bp on the 10Y par rate -> response of each forward bucket (bp)")
print("="*90)
print(f"{'segment':>10} {'Method A':>10} {'Method B':>10}")
for k in range(N):
    print(f"{SLBL[k]:>10} {Ja[k,c]:10.4f} {Jb[k,c]:10.4f}")

# ---- chart: instantaneous forward response to +1bp 10Y bump, both methods ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def inst_pw(t, dffun, h=1e-5):   # -d/dt ln DF, numeric
    return -(np.log(dffun(t+h)) - np.log(dffun(max(t-h,1e-9))))/(2*h)

pw_base = build_pw(PAR)
sp_base = build_sp(PAR)
pj = PAR.copy(); pj[c] += 1e-4
pw_bump = build_pw(pj)
sp_bump = build_sp(pj)

grid = np.linspace(0.1, 29.9, 400)
resp_pw = np.array([(inst_pw(t, pw_bump)-inst_pw(t, pw_base))/1e-4 for t in grid])
resp_sp = np.array([(inst_pw(t, sp_bump)-inst_pw(t, sp_base))/1e-4 for t in grid])

plt.figure(figsize=(10,5.2))
plt.step(grid, resp_pw, where='mid', lw=2, label='A: piecewise-constant forward (local)')
plt.plot(grid, resp_sp, lw=2, label='B: cubic spline on zeros (global)')
plt.axvline(10, color='grey', ls='--', lw=1)
plt.axhline(0, color='black', lw=0.6)
plt.title('Response of the instantaneous forward curve to a +1bp bump of the 10Y par rate')
plt.xlabel('maturity (years)'); plt.ylabel('forward response (bp per 1bp par bump)')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('/home/claude/jacobian_two_curves.png', dpi=130)
print("\nchart saved: jacobian_two_curves.png")

# sanity: both curves reprice the inputs
print("\nround-trip (worst reprice error):")
for nm, b in [('A pw', pw_base), ('B spline', sp_base)]:
    w = 0
    for i in range(N):
        times, accr = schedule(i)
        ann = sum(a*b(t) for t,a in zip(times,accr))
        w = max(w, abs(((1-b(T[i]))/ann - PAR[i])*1e4))
    print(f"   {nm:>9}: {w:.2e} bp")
