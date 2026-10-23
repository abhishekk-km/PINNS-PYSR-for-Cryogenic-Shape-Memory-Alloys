"""
02_landau_fit.py : Landau-Devonshire / Barrett / Wachtman analysis of the Ti-Al-Cr slope reversal.
Reads data/dataset.csv (override with SMA_DATA=path), writes outputs/landau/*. FAST=1 for a smoke test.
Model: f(eta) = a/2 eta^2 - b/4 eta^4 + c/6 eta^6 - sigma*eps*eta. The Maxwell construction collapses onto one
universal curve s(alpha); alpha(T) is driven by z(T) = A0*(T_Q(T) - T0) - K*n_B(T), with T_Q the Barrett
quantum-saturated temperature and n_B the Einstein phonon occupation of the elastic channel.
Also: digitisation Monte Carlo, kappa and Cp(T) calibration, conformal intervals, inter-model error table.
"""
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, least_squares, brentq
from scipy.interpolate import PchipInterpolator as PCHIP
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel, DotProduct
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
FAST = os.environ.get("FAST") == "1"; SEED = 42
DATA = os.environ.get("SMA_DATA", "data/dataset.csv"); OUT = "outputs/landau"; os.makedirs(OUT, exist_ok=True)
RHO, EPS = 4500., 0.064
NMC = 12 if FAST else 120; DE_IT = 60 if FAST else 250
rng = np.random.default_rng(SEED); R = {}; LOG = []
def P(s=""): print(s, flush=True); LOG.append(s)

df = pd.read_csv(DATA); ta = df[df.alloy_system == "Ti-Al-Cr"]
def ser(p, sp=None):
    s = ta[ta.property == p].dropna(subset=["T_K", "value"]).sort_values("T_K")
    if sp: s = s[s.split.isin(sp)]
    return s.T_K.values.astype(float), s.value.values.astype(float)
T13, S0 = ser("sigma_0", ["train", "test"]); _, SMS = ser("sigma_Ms", ["train", "test"])
Tq, Q = ser("Qdiss"); Ts, SE = ser("DeltaSeff"); Td, DS = ser("DeltaS_cal"); Tt, DT = ser("dTad_meas")
mape = lambda y, p: float(np.mean(np.abs(p - y) / np.abs(y)) * 100)
mae = lambda y, p: float(np.mean(np.abs(p - y)))

# L1  universal Maxwell curve
def _roots(alpha, s):
    r = np.roots([1, 0, -1, 0, alpha, -s]); r = np.sort(r[np.abs(r.imag) < 1e-7].real); return r[r > -1e-9]
def _Fy(y, a): return a * y**2 / 2 - y**4 / 4 + y**6 / 6
def _area(s, a):
    r = _roots(a, s)
    if len(r) < 3: return np.nan
    return _Fy(r[-1], a) - _Fy(r[0], a) - s * (r[-1] - r[0])
def maxwell(a):
    d = 9 - 20 * a
    if d <= 0: return np.nan, np.nan, np.nan
    ya, yb = np.sqrt((3 - np.sqrt(d)) / 10), np.sqrt((3 + np.sqrt(d)) / 10)
    h = lambda y: a * y - y**3 + y**5
    lo, hi = max(h(yb), 0.) + 1e-12, h(ya) - 1e-12
    if hi <= lo: return np.nan, np.nan, np.nan
    try: s = brentq(lambda t: _area(t, a), lo, hi, xtol=1e-14)
    except Exception: return np.nan, np.nan, np.nan
    r = _roots(a, s); return s, r[0], r[-1]
_g0 = lambda a: _area(1e-12, a)
AM = brentq(_g0, 1e-4, 0.2); AC = 0.4499
ALPHA = np.linspace(AM + 1e-6, AC, 500)
TAB = np.array([maxwell(a) for a in ALPHA]); STILDE, Y1, Y3 = TAB[:, 0], TAB[:, 1], TAB[:, 2]
ok = np.isfinite(STILDE); ALPHA, STILDE, Y1, Y3 = ALPHA[ok], STILDE[ok], Y1[ok], Y3[ok]
a_t = ALPHA[len(ALPHA) // 2]; s_t, y1, y3 = maxwell(a_t)       # self-test: equal minima, both stationary
g = lambda y: _Fy(y, a_t) - s_t * y
assert abs(g(y1) - g(y3)) < 1e-9 and abs(a_t * y1 - y1**3 + y1**5 - s_t) < 1e-9 and abs(a_t * y3 - y3**3 + y3**5 - s_t) < 1e-9
P(f"L1  Maxwell self-test passed. alpha_M={AM:.4f} (zero stress), alpha_C={AC} (critical); s_max={STILDE.max():.5f}")
R["maxwell"] = {"alpha_M": AM, "alpha_C": AC, "s_crit": float(STILDE.max())}

def barrett(T, Ts_): T = np.maximum(np.asarray(T, float), 1e-3); return 0.5 * Ts_ / np.tanh(Ts_ / (2 * T))
def nB(T, t): T = np.maximum(np.asarray(T, float), 1e-3); return 1. / np.expm1(np.minimum(t / T, 700.))
def zT(T, p): return p[1] * (barrett(T, p[3]) - p[2]) - p[4] * nB(T, p[5])
def alpha_z(z): return AM + (AC - AM) / (1 + np.exp(-np.clip(z, -50, 50)))
def sig_landau(T, p): return p[0] * np.interp(alpha_z(zT(T, p)), ALPHA, STILDE)
BND = [(1e3, 1e6), (1e-4, 0.1), (-100., 400.), (10., 300.), (0., 30.), (5., 300.)]
def fit_landau(T, y, seed=0, it=None):
    f = lambda p: np.sum(((sig_landau(T, p) - y) / 5.) ** 2)
    best = None
    for sd in (seed, seed + 1):
        r = differential_evolution(f, BND, seed=sd, maxiter=it or DE_IT, popsize=18, tol=1e-9, polish=False)
        if best is None or r.fun < best.fun: best = r
    lb, ub = zip(*BND)
    r = least_squares(lambda p: (sig_landau(T, p) - y) / 5., best.x, bounds=(lb, ub), max_nfev=400)
    return r.x if r.cost * 2 <= best.fun else best.x
Tg = np.linspace(20, 450, 431)
def argmin_T(c, T, lo=60, hi=300):
    m = (T >= lo) & (T <= hi); i = int(np.argmin(c[m])); return float(T[m][i]) if 0 < i < m.sum() - 1 else np.nan

PL = fit_landau(T13, S0, 0); fit_res = sig_landau(T13, PL) - S0
dz = np.gradient(zT(Tg, PL), Tg); Tinv_L = argmin_T(sig_landau(Tg, PL), Tg)
slope_lo = float(np.mean(np.gradient(sig_landau(Tg, PL), Tg)[(Tg >= 20) & (Tg <= 100)]))
slope_hi = float(np.mean(np.gradient(sig_landau(Tg, PL), Tg)[(Tg >= 250) & (Tg <= 400)]))
dzsign = Tg[:-1][(dz[:-1] < 0) & (dz[1:] >= 0)]
P(f"L1  Landau fit: params Sigma={PL[0]:.0f} A0={PL[1]:.4f} T0={PL[2]:.1f} Ts={PL[3]:.1f} K={PL[4]:.2f} tG={PL[5]:.1f}")
P(f"    in-sample MAPE {mape(S0, sig_landau(T13, PL)):.2f}%  MAE {mae(S0, sig_landau(T13, PL)):.2f} MPa | T_inv={Tinv_L:.1f} K | dz/dT sign change at {dzsign[:1]} K")
P(f"    slope 20-100 K {slope_lo:+.2f}  250-400 K {slope_hi:+.2f} MPa/K | slope at 20 K {np.gradient(sig_landau(Tg, PL), Tg)[0]:+.3f} (third law -> 0 as T->0)")
R["landau"] = {"params": PL.tolist(), "T_inv": Tinv_L, "MAPE_in_sample": mape(S0, sig_landau(T13, PL)), "slope_20_100": slope_lo, "slope_250_400": slope_hi,
               "dz_sign_change_K": dzsign.tolist()}
alpha_T = alpha_z(zT(Tg, PL)); eta_ratio = np.interp(alpha_T, ALPHA, Y3) - np.interp(alpha_T, ALPHA, Y1)
dSL = -(EPS / RHO) * 1e6 * np.gradient(sig_landau(Tg, PL), Tg)                # CC -> Landau entropy (J/kg/K)

# GP helpers + Barrett-GP + Landau-GP
def gp_fit(X, y, lin=True, restarts=3):
    sx, sy = StandardScaler(), StandardScaler(); Xs = sx.fit_transform(X.reshape(-1, 1)); z = sy.fit_transform(y.reshape(-1, 1)).ravel()
    k = C(1., (1e-2, 1e2)) * Matern(1., (.2, 10.), nu=2.5) + WhiteKernel(1e-3, (1e-6, 1e-1))
    if lin: k = C(1., (1e-2, 1e2)) * DotProduct(sigma_0=1., sigma_0_bounds=(1e-2, 1e2)) + k
    gp = GPR(k, n_restarts_optimizer=restarts, alpha=1e-8, random_state=SEED).fit(Xs, z)
    def f(Xn, sd=False):
        zz, s = gp.predict(sx.transform(np.atleast_1d(Xn).reshape(-1, 1)), return_std=True)
        m = sy.inverse_transform(zz.reshape(-1, 1)).ravel(); return (m, s * sy.scale_[0]) if sd else m
    f.lml = gp.log_marginal_likelihood_value_; return f
def fit_plain(T, y): m = gp_fit(T, y); return lambda t, sd=False: m(t, sd)
def fit_bgp(T, y, grid=(15., 40., 80.)):
    best = max((gp_fit(barrett(T, ts), y) for ts in grid), key=lambda m: m.lml); ts = grid[int(np.argmax([gp_fit(barrett(T, t), y).lml for t in grid]))]
    return lambda t, sd=False: best(barrett(t, ts), sd)
def fit_lgp(T, y, it=None):
    p = fit_landau(T, y, 0, it); r = gp_fit(T, y - sig_landau(T, p), lin=False)
    return lambda t, sd=False: (sig_landau(t, p) + r(t)) if not sd else (sig_landau(t, p) + r(t, True)[0], r(t, True)[1])
def fit_land(T, y, it=None): p = fit_landau(T, y, 0, it); return lambda t, sd=False: sig_landau(t, p)
FIT = {"GP": fit_plain, "Barrett-GP": fit_bgp, "Landau": lambda T, y: fit_land(T, y, DE_IT // 2), "Landau-GP": lambda T, y: fit_lgp(T, y, DE_IT // 2)}

P("\nL3  LOO (sigma_0), mean absolute percentage error and MAE")
loo = {}
for nm, fn in FIT.items():
    pr = np.array([float(fn(T13[np.arange(len(T13)) != i], S0[np.arange(len(T13)) != i])(T13[i:i + 1])[0]) for i in range(len(T13))])
    loo[nm] = {"MAPE": mape(S0, pr), "MAE": mae(S0, pr), "pred": pr.tolist()}; P(f"  {nm:11s} MAPE {loo[nm]['MAPE']:.2f}%  MAE {loo[nm]['MAE']:.2f} MPa")
R["loo"] = loo
P("\nL3  Extrapolation from higher-temperature data (train T>=cut), held points below the cut")
ext = {}
for cut in (150., 100.):
    m = T13 >= cut; held = ~m
    if held.sum() < 2: continue
    obs = np.polyfit(T13[held], S0[held], 1)[0]
    for nm, fn in FIT.items():
        mdl = fn(T13[m], S0[m]); pr = mdl(T13[held]); tt = np.linspace(20, cut, 60); sl = float(np.polyfit(tt, mdl(tt), 1)[0])
        ext[f"{nm}@{cut:.0f}"] = {"MAPE_held": mape(S0[held], pr), "slope_pred": sl, "slope_obs": float(obs), "sign_ok": bool(sl * obs > 0)}
        P(f"  cut {cut:.0f} K {nm:11s} MAPE {mape(S0[held], pr):6.1f}%  slope pred {sl:+.2f} obs {obs:+.2f} sign_ok={sl * obs > 0}")
R["extrapolation"] = ext

# L2  digitisation Monte Carlo
P(f"\nL2  Digitisation Monte Carlo ({NMC} draws: T +-3 K, stress +-2 MPa)")
inv_L, inv_B, curv_L = [], [], []
for k in range(NMC):
    Tj = T13 + rng.normal(0, 3, T13.size); yj = S0 + rng.normal(0, 2, T13.size); o = np.argsort(Tj); Tj, yj = Tj[o], yj[o]
    p = fit_landau(Tj, yj, k, DE_IT // 3); c = sig_landau(Tg, p); inv_L.append(argmin_T(c, Tg)); curv_L.append(c)
    b = fit_bgp(Tj, yj); inv_B.append(argmin_T(b(Tg), Tg))
ci = lambda a: [float(np.nanmedian(a)), *np.nanpercentile(a, [2.5, 97.5]).tolist()]
R["mc_inversion"] = {"Landau": ci(inv_L), "Barrett-GP": ci(inv_B)}; curv_L = np.array(curv_L)
P(f"  T_inv Landau median {R['mc_inversion']['Landau'][0]:.1f} K  95% [{R['mc_inversion']['Landau'][1]:.1f}, {R['mc_inversion']['Landau'][2]:.1f}]")
P(f"  T_inv Barrett-GP median {R['mc_inversion']['Barrett-GP'][0]:.1f} K  95% [{R['mc_inversion']['Barrett-GP'][1]:.1f}, {R['mc_inversion']['Barrett-GP'][2]:.1f}]")

# L4  caloric mitigations
Ti = np.arange(max(Ts.min(), Td.min(), Tq.min()), min(Ts.max(), Td.max(), Tq.max()) + 1, 2.)
dSi, Qi, Sei = PCHIP(Td, DS)(Ti), PCHIP(Tq, Q)(Ti), PCHIP(Ts, SE)(Ti); reg = Qi / (2 * Ti); res = Sei - dSi
kap = float((res * reg).sum() / (reg * reg).sum()); bs = []
for _ in range(1000):
    i = rng.integers(0, len(Ti), len(Ti)); bs.append((res[i] * reg[i]).sum() / (reg[i] ** 2).sum())
zc = lambda c, T: (lambda i: float(T[i[0]] + (T[i[0] + 1] - T[i[0]]) * c[i[0]] / (c[i[0]] - c[i[0] + 1])) if i.size else np.nan)(np.where((c[:-1] > 0) & (c[1:] <= 0))[0])
Tmin_k = zc(dSi + kap * reg, Ti); Tmin_1 = zc(dSi + reg, Ti)
P(f"\nL4  kappa = {kap:.3f} 95% [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}] | T_min kappa=1: {Tmin_1:.1f} K, kappa fitted: {Tmin_k:.1f} K, measured Seff: {zc(Sei, Ti):.1f} K")
R["kappa"] = {"fit": kap, "CI95": np.percentile(bs, [2.5, 97.5]).tolist(), "Tmin_kappa1": Tmin_1, "Tmin_kappa_fit": Tmin_k}
hi = (T13 >= 250); slope_m = np.polyfit(T13[hi], S0[hi], 1)[0]; Th = Ti[Ti >= 250]
imp = np.mean(-(RHO / EPS) * 1e-6 * PCHIP(Td, DS)(Th)); fct = slope_m / imp; fb = []
for _ in range(1000):
    i = rng.integers(0, hi.sum(), hi.sum())
    if len(np.unique(T13[hi][i])) > 1: fb.append(np.polyfit(T13[hi][i], S0[hi][i], 1)[0] / imp)
P(f"    rho/eps_tr scale factor f = {fct:.3f} 95% [{np.percentile(fb, 2.5):.3f}, {np.percentile(fb, 97.5):.3f}] (f=1: rho=4500, eps=0.064 consistent above 250 K)")
R["conversion_factor"] = {"f": float(fct), "CI95": np.percentile(fb, [2.5, 97.5]).tolist()}

_x = np.linspace(1e-6, 60, 60001); _gg = _x**4 * np.exp(_x) / np.expm1(_x)**2; _I = np.concatenate([[0], np.cumsum(.5 * (_gg[1:] + _gg[:-1]) * np.diff(_x))])
def debye(t): t = np.maximum(np.asarray(t, float), 1e-6); return 3 * t**3 * np.interp(1. / t, _x, _I)
A_DP = 3 * 8.314462618 / ((0.7525 * 47.867 + 0.20 * 26.982 + 0.0475 * 51.996) / 1000)
cp_m = lambda T, g_, th: g_ * T + A_DP * debye(T / th)
ok_ = np.abs(DT) >= 1.; Tc, Cpi = Tt[ok_], Tt[ok_] * np.interp(Tt[ok_], Ts, SE) / DT[ok_]
def fit_cp(T, c):
    r = [least_squares(lambda p: cp_m(T, *p) - c, [g0, th0], bounds=([0, 50], [1.5, 900])) for th0 in (150, 300, 500) for g0 in (.01, .1)]
    return min(r, key=lambda z: z.cost).x
pcp = fit_cp(Tc, Cpi); Sint = lambda t: np.interp(t, Ts, SE)
loo_d, loo_k = [], []
for i in range(len(Tc)):
    m = np.arange(len(Tc)) != i; pp = fit_cp(Tc[m], Cpi[m]); loo_d.append(Tc[i] * Sint(Tc[i]) / cp_m(Tc[i], *pp)); loo_k.append(Tc[i] * Sint(Tc[i]) / 588.)
meas = DT[ok_]; sel = np.abs(meas) >= 2
P(f"    Debye Cp: gamma={pcp[0]:.4f} theta={pcp[1]:.0f} K | LOO dTad (|dT|>=2 K): Debye MAPE {mape(meas[sel], np.array(loo_d)[sel]):.1f}% MAE {mae(meas, np.array(loo_d)):.2f} K ; constant-588 MAPE {mape(meas[sel], np.array(loo_k)[sel]):.1f}% MAE {mae(meas, np.array(loo_k)):.2f} K")
R["debye_cp"] = {"gamma": float(pcp[0]), "theta": float(pcp[1]), "LOO_MAPE_debye": mape(meas[sel], np.array(loo_d)[sel]), "LOO_MAE_debye": mae(meas, np.array(loo_d)),
                 "LOO_MAPE_const588": mape(meas[sel], np.array(loo_k)[sel]), "LOO_MAE_const588": mae(meas, np.array(loo_k))}

Tx = np.array([50., 100., 150., 200., 300., 400.]); exs = {}
P("    Landau excess entropy  dS_cal - dS_Landau(CC)  (J/kg/K) and heat-equivalent T*excess (J/kg)")
for t in Tx:
    if Td.min() <= t <= Td.max():
        e = float(np.interp(t, Td, DS) - np.interp(t, Tg, dSL)); exs[int(t)] = [e, t * e]; P(f"      {t:4.0f} K  {e:+7.2f}   {t * e:+9.0f}")
R["excess_entropy"] = exs

Z_ = []
for nm, tr in (("Barrett-GP", fit_bgp), ("GP", fit_plain)):
    zz = []
    for i in range(len(T13)):
        m = np.arange(len(T13)) != i; mu, sd = tr(T13[m], S0[m])(T13[i:i + 1], True); zz.append(abs(S0[i] - mu[0]) / sd[0])
    zz = np.array(zz); q = float(np.quantile(zz, min(1., np.ceil((len(zz) + 1) * .95) / len(zz))))
    Z_.append((nm, float(np.mean(zz < 1.96)), q / 1.96)); P(f"    conformal {nm:10s}: PICP(1.96)={np.mean(zz < 1.96):.2f}, inflation x{q / 1.96:.2f} for 95% LOO coverage")
R["conformal"] = {n: {"PICP": a, "inflation": b} for n, a, b in Z_}

# L5  inter-model table
tst = ta[(ta.property == "sigma_Ms") & (ta.split == "test")].sort_values("T_K"); Tte, yte = tst.T_K.values.astype(float), tst.value.values.astype(float)
trn = ta[(ta.property == "sigma_Ms") & (ta.split == "train")]; Ttr, ytr = trn.T_K.values.astype(float), trn.value.values.astype(float)
IM = {}
if len(Tte):
    for nm, fn in FIT.items(): IM[nm] = mape(yte, fn(Ttr, ytr)(Tte))
    try:
        import sympy as sp
        pr = pd.read_csv("outputs/pysr_equations_pareto.csv"); row = pr.loc[pr["score"].idxmax()]; x0 = sp.Symbol("x0")
        ex = sp.sympify(str(row["equation"]), locals={"square": lambda z: z**2}); f = sp.lambdify(x0, ex, "numpy")
        IM["PySR"] = mape(yte, f(Tte / 100.) * ytr.std() + ytr.mean()); R["pysr_equation"] = str(row["equation"])
    except Exception as e: P(f"  PySR comparison skipped: {type(e).__name__}")
    try:
        pp = pd.read_csv("outputs/pinn_v6_predictions.csv").sort_values("T_K"); IM["PINN"] = mape(yte, np.interp(Tte, pp.T_K, pp.sigma_Ms))
    except Exception as e: P(f"  PINN comparison skipped: {type(e).__name__}")
P("\nL5  Held-out sigma_Ms MAPE by model: " + ", ".join(f"{k} {v:.2f}%" for k, v in IM.items())); R["intermodel_MAPE_sigmaMs_heldout"] = IM
inv_ref = {"GP": R["mc_inversion"]["Barrett-GP"][0], "Landau": Tinv_L}; R["inversion_deviation_pct_from_150"] = {k: abs(v - 150) / 150 * 100 for k, v in inv_ref.items()}

# figure
fig, ax = plt.subplots(2, 3, figsize=(16, 9)); a = ax[0, 0]
a.fill_between(Tg, *np.nanpercentile(curv_L, [2.5, 97.5], 0), color="C0", alpha=.2, label="Landau, 95% digitisation MC")
a.plot(Tg, sig_landau(Tg, PL), "C0", lw=2, label="Landau (Maxwell construction)"); a.scatter(T13, S0, c="k", zorder=5, label="data")
a.axvline(Tinv_L, c="C3", ls="--", label=f"T_inv {Tinv_L:.0f} K"); a.set(xlabel="T (K)", ylabel="sigma_0 (MPa)", title="(a) Landau fit"); a.legend(fontsize=7)
a = ax[0, 1]; a.plot(Tg, zT(Tg, PL), "C1"); a2 = a.twinx(); a2.plot(Tg, dz, "C3"); a2.axhline(0, c="k", lw=.7)
a.set(xlabel="T (K)", ylabel="z(T)  (Landau coefficient)", title="(b) Landau coefficient and sign change of dz/dT"); a2.set_ylabel("dz/dT", color="C3")
a = ax[0, 2]; a.plot(ALPHA, STILDE, "C2"); a.set(xlabel="alpha = a/b^2", ylabel="s(alpha)", title="(c) Universal Maxwell curve (2-4-6 potential)")
a = ax[1, 0]; a.plot(Tg, dSL, "C0", label="Landau via CC"); a.scatter(Td, DS, c="k", s=18, label="calorimetric"); a.axhline(0, c="k", lw=.6)
a.set(xlabel="T (K)", ylabel="Delta S (J/kg/K)", title="(d) Landau entropy vs calorimetry"); a.legend(fontsize=7)
a = ax[1, 1]; a.plot(Ti, dSi + reg, "C1", label="kappa=1"); a.plot(Ti, dSi + kap * reg, "C3", label=f"kappa={kap:.2f}"); a.plot(Ts, SE, "ko", ms=3, label="measured")
a.axhline(0, c="k", lw=.6); a.set(xlabel="T (K)", ylabel="Seff (J/kg/K)", title="(e) kappa calibration"); a.legend(fontsize=7)
a = ax[1, 2]; names = [n for n in IM if n != "PINN"]; a.bar(range(len(names)), [IM[n] for n in names], color="C0"); a.set_xticks(range(len(names))); a.set_xticklabels(names, rotation=30, ha="right")
a.set(ylabel="held-out MAPE (%)", title="(f) Inter-model stress error"); plt.tight_layout(); plt.savefig(f"{OUT}/fig_landau.png", dpi=200); plt.close()
os.makedirs("outputs/paper", exist_ok=True)
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2)); ax[0].bar(range(len(names)), [IM[n] for n in names]); ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names, rotation=30, ha="right"); ax[0].set(ylabel="held-out MAPE (%)", title="(a) stress error")
iv = {"GP": R["mc_inversion"]["Barrett-GP"][0], "Landau": Tinv_L}; ax[1].bar(list(iv), list(iv.values())); ax[1].axhline(150, c="r", ls="--", label="experiment 150 K"); ax[1].legend(); ax[1].set(ylabel="T_inv (K)", title="(b) slope reversal")
ax[2].bar(["Tmin kappa=1", "Tmin kappa fit"], [abs(Tmin_1 - 97) / 97 * 100, abs(Tmin_k - 97) / 97 * 100]); ax[2].set(ylabel="deviation from 97 K (%)", title="(c) cooling onset")
plt.tight_layout(); plt.savefig("outputs/paper/fig_intermodel.png", dpi=200); plt.close()
json.dump(R, open(f"{OUT}/paper_metrics.json", "w"), indent=1, default=float); json.dump({"params": PL.tolist(), "bounds": BND}, open(f"{OUT}/landau_params.json", "w"))
open(f"{OUT}/summary.txt", "w").write("\n".join(LOG)); P(f"\nSaved {OUT}/paper_metrics.json, landau_params.json, fig_landau.png, outputs/paper/fig_intermodel.png")
np.savez(f"{OUT}/maxwell_table.npz", ALPHA=ALPHA, STILDE=STILDE, AM=AM, AC=AC)
# Physics: the stress slope is the sum of a chemical channel (heat-carrying, visible in calorimetry down to 48 K) and an
# elastic channel (phonon-activated shear-modulus term). Calorimetry is an INDEPENDENT observable that exists below the stress
# cutoff, so a joint fit constrains the chemical channel everywhere and the stress data above the cutoff fix the elastic channel.
def sig_chem(T, p): return p[0] * np.interp(alpha_z(p[1] * (barrett(T, p[3]) - p[2])), ALPHA, STILDE)
def dS_chem(T, p, h=0.5): T = np.asarray(T, float); return -p[6] * (EPS / RHO) * 1e6 * (sig_chem(T + h, p) - sig_chem(T - h, p)) / (2 * h)
BND7 = BND + [(0.3, 1.5)]
def fit_joint(T, y, it=None, seed=0):
    f = lambda p: np.sum(((sig_landau(T, p[:6]) - y) / 5.) ** 2) + np.sum(((dS_chem(Td, p) - DS) / 1.5) ** 2)
    best = min((differential_evolution(f, BND7, seed=s, maxiter=it or DE_IT, popsize=18, tol=1e-9, polish=False) for s in (seed, seed + 1)), key=lambda r: r.fun)
    lb, ub = zip(*BND7)
    r = least_squares(lambda p: np.r_[(sig_landau(T, p[:6]) - y) / 5., (dS_chem(Td, p) - DS) / 1.5], best.x, bounds=(lb, ub), max_nfev=400)
    return r.x if 2 * r.cost <= best.fun else best.x
P("\nL6  Recovery horizon: largest data cutoff at which each model recovers the sign and size of the low-temperature branch")
HZ = {}
for cut in (200., 170., 150., 130., 120., 110., 100., 80.):
    m = T13 >= cut; held = ~m
    if held.sum() < 2 or m.sum() < 5: continue
    obs = np.polyfit(T13[held], S0[held], 1)[0]; tt = np.linspace(20, cut, 60); row = {}
    gp = fit_plain(T13[m], S0[m]); lg = fit_lgp(T13[m], S0[m], DE_IT // 2); pj = fit_joint(T13[m], S0[m], DE_IT // 2)
    for nm, mdl in (("GP", gp), ("Landau-GP", lg), ("Joint-Landau", lambda t, p=pj: sig_landau(t, p[:6]))):
        sl = float(np.polyfit(tt, mdl(tt), 1)[0]); row[nm] = {"MAPE": mape(S0[held], mdl(T13[held])), "slope_pred": sl, "sign_ok": bool(sl * obs > 0)}
    HZ[f"{cut:.0f}"] = {"slope_obs": float(obs), **row}
    P(f"  cut {cut:3.0f} K (obs slope {obs:+.2f}) | " + " | ".join(f"{k} {v['MAPE']:5.1f}% {v['slope_pred']:+.2f} {'OK' if v['sign_ok'] else '--'}" for k, v in row.items()))
R["recovery_horizon"] = HZ
json.dump(HZ, open(f"{OUT}/extrapolation_fix.json", "w"), indent=1, default=float)
P(f"Saved {OUT}/extrapolation_fix.json and {OUT}/maxwell_table.npz")
