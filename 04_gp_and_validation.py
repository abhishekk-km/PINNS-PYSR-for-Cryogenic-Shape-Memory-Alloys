"""
04_gp_and_validation.py
Physics-informed surrogate analysis of anomalous Clausius-Clapeyron behaviour in
Ti-Al-Cr (M1 Gaussian process, M2 PINN, M3 symbolic regression).
Reads data/dataset.csv. Writes everything to outputs/final/.
Run:  python 04_gp_and_validation.py        (FAST=1 python 04_gp_and_validation.py for a smoke test)
"""
import os, json, pickle, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from scipy.stats import wilcoxon
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import (Matern, ConstantKernel as C,
                                              WhiteKernel, DotProduct)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import torch, torch.nn as nn

warnings.filterwarnings("ignore")
FAST = os.environ.get("FAST") == "1"
EPOCHS, N_CV, N_FIN, PY_IT = (800, 2, 3, 40) if FAST else (3000, 3, 5, 150)
SEED = 42
OUT = "outputs/final"; os.makedirs(OUT, exist_ok=True)
EPS, RHO, CPV = 0.064, 4500.0, 588.0
np.random.seed(SEED); torch.manual_seed(SEED)
R = {}

PAL = dict(gp="#1f6fb2", pinn="#d1495b", pysr="#2a9d8f", pc="#8d99ae", dat="#111111", inv="#6a4c93")
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "figure.dpi": 120,
                     "axes.titleweight": "bold", "legend.frameon": False, "legend.fontsize": 8})

def zones(a):
    a.axvspan(0, 50, color="#d1495b", alpha=.07, lw=0)
    a.axvspan(50, 150, color="#f4a261", alpha=.08, lw=0)

def band(a, x, lo, hi, c, lab=None):
    a.fill_between(x, lo, hi, color=c, alpha=.2, lw=0, label=lab)

# data
df = pd.read_csv("data/dataset.csv")
df["T_K"] = pd.to_numeric(df["T_K"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")
ta = df[df.alloy_system == "Ti-Al-Cr"]

def series(prop, splits=None):
    s = ta[ta.property == prop]
    if splits: s = s[s.split.isin(splits)]
    s = s.dropna(subset=["T_K", "value"]).sort_values("T_K")
    return s["T_K"].values.astype(float), s["value"].values.astype(float)

T13, s0 = series("sigma_0", ["train", "test"])
_, sMs = series("sigma_Ms", ["train", "test"]); _, sAf = series("sigma_Af", ["train", "test"])
Tq, Q = series("Qdiss"); Ts, S = series("DeltaSeff"); Tc, dTm = series("dTad_meas")
Tk, Sc = series("DeltaS_cal")
assert len(T13) == len(sMs) == len(sAf) == 13
SER = {"sigma_0": (T13, s0), "sigma_Ms": (T13, sMs), "sigma_Af": (T13, sAf), "Qdiss": (Tq, Q),
       "DeltaSeff": (Ts, S), "DeltaS_cal": (Tk, Sc), "dTad": (Tc, dTm)}
R["n_stress"] = 13

def metr(y, p):
    return {"R2": float(r2_score(y, p)), "MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(np.mean((y - p) ** 2)))}

# data audit: CC and Song relation
Tg2 = np.arange(200, 401, 10.)
gap = np.interp(Tg2, Ts, S) - np.interp(Tg2, Tk, Sc)
reg = np.interp(Tg2, Tq, Q) / (2 * Tg2)
KAPPA = float((gap * reg).sum() / (reg * reg).sum())
R["song_audit"] = {"kappa_fit": KAPPA, "mean_abs_residual_kappa1_JkgK": float(np.mean(np.abs(gap - reg))),
                   "mean_abs_residual_kappa_fit_JkgK": float(np.mean(np.abs(gap - KAPPA * reg)))}
hi = T13 >= 200
R["cc_audit_classical"] = {"measured_slope_MPaK": float(np.polyfit(T13[hi], s0[hi], 1)[0]),
                           "cc_implied_slope_MPaK": float(np.mean(-np.interp(Tg2, Tk, Sc) * RHO / EPS / 1e6))}

# GP helpers
def fit_stress_gp(T, y, restarts=10):
    sx, sy = StandardScaler(), StandardScaler()
    X = sx.fit_transform(T.reshape(-1, 1)); z = sy.fit_transform(y.reshape(-1, 1)).ravel()
    k = (C(1., (1e-2, 1e2)) * DotProduct(sigma_0=1., sigma_0_bounds=(1e-2, 1e2))
         + C(.5, (1e-2, 1e2)) * Matern(1., (.2, 10.), nu=2.5) + WhiteKernel(1e-3, (1e-6, 1e-1)))
    return GPR(k, n_restarts_optimizer=restarts, alpha=1e-8, random_state=SEED).fit(X, z), sx, sy

def fit_cal_gp(T, y, restarts=10):
    sx, sy = StandardScaler(), StandardScaler()
    X = sx.fit_transform(T.reshape(-1, 1)); z = sy.fit_transform(y.reshape(-1, 1)).ravel()
    k = C(1., (1e-2, 1e2)) * Matern(1., (.2, 10.), nu=2.5) + WhiteKernel(1e-3, (1e-6, 1e-1))
    return GPR(k, n_restarts_optimizer=restarts, alpha=1e-8, random_state=SEED).fit(X, z), sx, sy

def pred(m, T):
    g, sx, sy = m
    z, s = g.predict(sx.transform(np.atleast_1d(T).reshape(-1, 1)), return_std=True)
    return sy.inverse_transform(z.reshape(-1, 1)).ravel(), s * sy.scale_[0]

def draws(m, T, n=1000, seed=0):
    g, sx, sy = m
    z = g.sample_y(sx.transform(np.atleast_1d(T).reshape(-1, 1)), n_samples=n, random_state=seed)
    return z * sy.scale_[0] + sy.mean_[0]

def first_inversion(curve, T, lo=60, hi=300):
    d = np.gradient(curve, T)
    i = np.where((T[:-1] >= lo) & (T[:-1] <= hi) & (d[:-1] < 0) & (d[1:] >= 0))[0]
    return float(.5 * (T[i[0]] + T[i[0] + 1])) if i.size else np.nan

def argmin_T(curve, T, lo=80, hi=300):
    m = (T >= lo) & (T <= hi); i = int(np.argmin(curve[m])); Tm = T[m]
    return float(Tm[i]) if 0 < i < m.sum() - 1 else np.nan

def zero_cross(curve, T):
    i = np.where((curve[:-1] > 0) & (curve[1:] <= 0))[0]
    if not i.size: return np.nan
    i = i[0]; return float(T[i] + (T[i + 1] - T[i]) * curve[i] / (curve[i] - curve[i + 1]))

# M1: GAUSSIAN PROCESS
print("M1: Gaussian process")
def loo_pred(T, y, kind):
    out, sd = [], []
    for i in range(len(T)):
        m = np.ones(len(T), bool); m[i] = False
        if kind == "GP":
            g = fit_stress_gp(T[m], y[m], 5); p, s = pred(g, T[i]); out.append(p[0]); sd.append(s[0])
        elif kind == "PCHIP":
            out.append(float(PchipInterpolator(T[m], y[m], extrapolate=True)(T[i])))
        else:
            out.append(float(np.polyval(np.polyfit(T[m], y[m], int(kind[-1])), T[i])))
    return np.array(out), np.array(sd)

stress_loo, loo_err = {}, {}
for nm, y in [("sigma_0", s0), ("sigma_Ms", sMs), ("sigma_Af", sAf)]:
    rows = {}
    for kind in ["GP", "PCHIP", "poly1", "poly2", "poly3"]:
        p, sd = loo_pred(T13, y, kind)
        rows[kind] = metr(y, p); loo_err[(nm, kind)] = y - p
        if kind == "GP":
            rows[kind]["PICP95"] = float(np.mean(np.abs(y - p) / sd < 1.96))
            if nm == "sigma_0": R["loo_pred_sigma_0_GP"] = p.tolist()
    stress_loo[nm] = rows
R["stress_loo"] = stress_loo
try:
    R["wilcoxon_GP_vs_PCHIP_p"] = float(wilcoxon(np.abs(loo_err[("sigma_0", "GP")]),
                                                  np.abs(loo_err[("sigma_0", "PCHIP")])).pvalue)
except Exception:
    R["wilcoxon_GP_vs_PCHIP_p"] = None

G0 = fit_stress_gp(T13, s0, 20); Tg = np.linspace(20, 450, 431)
mean0, sd0 = pred(G0, Tg); S0 = draws(G0, Tg, seed=1)
d_mean = np.gradient(mean0, Tg); dS0 = np.gradient(S0, Tg, axis=0)
Tinv_s = np.array([argmin_T(S0[:, j], Tg) for j in range(S0.shape[1])]); ok = ~np.isnan(Tinv_s)
R["inversion_GP"] = {"mean_curve_K": first_inversion(mean0, Tg), "posterior_frac": float(ok.mean()),
                     "posterior_median_K": float(np.median(Tinv_s[ok])),
                     "posterior_CI95_K": [float(np.percentile(Tinv_s[ok], 2.5)), float(np.percentile(Tinv_s[ok], 97.5))],
                     "slope_20_100": float(d_mean[(Tg >= 20) & (Tg <= 100)].mean()),
                     "slope_250_400": float(d_mean[(Tg >= 250) & (Tg <= 400)].mean())}
fd = np.diff(s0) / np.diff(T13); Tmid = .5 * (T13[:-1] + T13[1:])
R["data_negative_slope_interval_K"] = [float(Tmid[fd < 0].min()), float(Tmid[fd < 0].max())]

# CC violation with posterior intervals
GSC = fit_cal_gp(Tk, Sc, 20)
Tv = np.linspace(50, 400, 351)
SV = np.gradient(draws(G0, Tv, seed=11), Tv, axis=0)
CCV = -draws(GSC, Tv, seed=12) * RHO / EPS / 1e6
EX = SV - CCV
def win(lo, hi):
    w = (Tv >= lo) & (Tv <= hi); x = EX[w].mean(0)
    return {"mean": float(x.mean()), "CI95": [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))],
            "P_negative": float((x < 0).mean())}
R["cc_violation"] = {"window_50_150": win(50, 150), "window_250_400": win(250, 400)}

def gap_test(lo, hi, T=T13, y=s0):
    m = (T < lo) | (T > hi); g = fit_stress_gp(T[m], y[m], 10); mu, sd = pred(g, Tg)
    ph, sh = pred(g, T[~m])
    return {"held_T": T[~m].tolist(), "held_MAE": float(np.mean(np.abs(ph - y[~m]))),
            "inside_95CI": [bool(abs(a - b) < 1.96 * c) for a, b, c in zip(ph, y[~m], sh)],
            "T_inv_K": first_inversion(mu, Tg)}
R["gap_test_GP"] = gap_test(100, 200)
mx = T13 >= 150
gx = fit_stress_gp(T13[mx], s0[mx], 10); Tx = np.linspace(20, 150, 131); mux, _ = pred(gx, T13[~mx])
R["extrapolation_GP"] = {"held_MAE": float(np.mean(np.abs(mux - s0[~mx]))),
                         "pred_slope": float(np.gradient(pred(gx, Tx)[0], Tx).mean()),
                         "obs_slope": float(np.polyfit(T13[~mx], s0[~mx], 1)[0])}
R["extrapolation_GP"]["sign_recovered"] = bool(R["extrapolation_GP"]["pred_slope"] < 0)

GQ = fit_cal_gp(Tq, Q, 20); GS = fit_cal_gp(Ts, S, 20)
cal = {}
for nm, T, y in [("Qdiss", Tq, Q), ("DeltaSeff", Ts, S), ("dTad_meas", Tc, dTm)]:
    p = np.zeros(len(T)); pc = np.zeros(len(T)); sd = np.zeros(len(T))
    for i in range(len(T)):
        m = np.ones(len(T), bool); m[i] = False
        a, b = pred(fit_cal_gp(T[m], y[m], 5), T[i]); p[i], sd[i] = a[0], b[0]
        pc[i] = float(PchipInterpolator(T[m], y[m], extrapolate=True)(T[i]))
    cal[nm] = {"n": len(T), "GP": metr(y, p), "PCHIP": metr(y, pc),
               "PICP95": float(np.mean(np.abs(y - p) / sd < 1.96))}
R["caloric_loo"] = cal

Tz = np.linspace(Ts.min(), Ts.max(), 1000); Sd = draws(GS, Tz, seed=2)
zc = np.array([zero_cross(Sd[:, j], Tz) for j in range(Sd.shape[1])]); zc = zc[~np.isnan(zc)]
R["Tmin_GP"] = {"median": float(np.median(zc)), "CI95": [float(np.percentile(zc, 2.5)), float(np.percentile(zc, 97.5))],
                "data_linear": zero_cross(S, Ts), "Song": 97.0}
Tcop = np.linspace(110, min(Tq.max(), Ts.max()), 300)
COPd = Tcop[:, None] * np.abs(draws(GS, Tcop, seed=4)) / draws(GQ, Tcop, seed=3)
COPd[draws(GS, Tcop, seed=4) >= 0] = np.nan
cop_med = np.nanmedian(COPd, 1); cop_lo = np.nanpercentile(COPd, 2.5, 1); cop_hi = np.nanpercentile(COPd, 97.5, 1)
i294 = int(np.argmin(np.abs(Tcop - 294)))
R["COP_GP"] = {"at_294": float(cop_med[i294]), "CI95": [float(cop_lo[i294]), float(cop_hi[i294])], "Song": 4.6,
               "raw_arithmetic": float(294 * abs(np.interp(294, Ts, S)) / np.interp(294, Tq, Q)),
               "peak": float(np.nanmax(cop_med)), "peak_T": float(Tcop[np.nanargmax(cop_med)])}
Sint = np.interp(Tc, Ts, S); sel = Tc >= 200
cpi = Tc[sel] * Sint[sel] / dTm[sel]
R["Cp_implied_ge200"] = {"mean": float(cpi.mean()), "sd": float(cpi.std(ddof=1)), "n": int(sel.sum()), "lit": 588.0}

# M2: PINN
print("M2: PINN")
MU = torch.tensor([400., 250., 1300., -20., -5., 590.]); SCL = torch.tensor([200., 150., 300., 15., 15., 60.])
IDX = {"sigma_Ms": 0, "sigma_Af": 1, "Qdiss": 2, "DeltaS_cal": 3, "DeltaSeff": 4}
W_CC, W_SG = 0.02, 0.05

def tx(T): return torch.tensor((np.asarray(T, float) - 225.) / 225., dtype=torch.float32).reshape(-1, 1)
TCOL = np.linspace(200, 402, 60); XC = tx(TCOL); TCT = torch.tensor(TCOL, dtype=torch.float32)

class Net(nn.Module):
    def __init__(s, h=48):
        super().__init__()
        s.f = nn.Sequential(nn.Linear(1, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 6))
    def forward(s, x): return s.f(x)

def train_pinn(tr, phys, seed, epochs=EPOCHS, track=False):
    torch.manual_seed(seed)
    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-4)
    sup = [(IDX[k], tx(T), torch.tensor(y, dtype=torch.float32)) for k, (T, y) in tr.items() if k in IDX]
    dt = None
    if "dTad" in tr:
        Td, yd = tr["dTad"]; m = Td >= 140
        if m.any(): dt = (tx(Td[m]), torch.tensor(Td[m], dtype=torch.float32), torch.tensor(yd[m], dtype=torch.float32))
    hist, ep0 = [], epochs // 6
    for ep in range(epochs):
        opt.zero_grad(); Ld = torch.zeros(())
        for j, x, y in sup:
            Ld = Ld + ((net(x)[:, j] - (y - MU[j]) / SCL[j]) ** 2).mean()
        if dt is not None:
            x, Tt, yt = dt; ph = net(x) * SCL + MU
            Ld = Ld + (((Tt * ph[:, 4] / ph[:, 5] - yt) / 5.) ** 2).mean()
        Lp = torch.zeros(())
        if phys and ep >= ep0:
            w = min(1., (ep - ep0) / ep0)
            xc = XC.clone().requires_grad_(True); ph = net(xc) * SCL + MU
            d = torch.autograd.grad((0.5 * (ph[:, 0] + ph[:, 1])).sum(), xc, create_graph=True)[0][:, 0] / 225.
            cc = -ph[:, 3] * RHO / EPS / 1e6
            rcc = torch.relu((d - cc).abs() - 0.25 * cc.abs())
            rsg = ph[:, 4] - ph[:, 3] - KAPPA * ph[:, 2] / (2 * TCT)
            Lp = w * (W_CC * (rcc ** 2).mean() + W_SG * ((rsg / 5.) ** 2).mean())
        loss = Ld + Lp; loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5.); opt.step(); sch.step()
        if track: hist.append((loss.item(), float(Ld), float(Lp)))
    return net, hist

def members(nets, T):
    x = tx(T)
    with torch.no_grad():
        return np.stack([(n(x) * SCL + MU).numpy() for n in nets])

def ser_pred(a, T, name):
    if name == "sigma_0": return 0.5 * (a[..., 0] + a[..., 1])
    if name == "dTad": return np.asarray(T)[None, :] * a[..., 4] / a[..., 5]
    return a[..., IDX[name]]

# shared 4-fold interleaved CV: GP, PCHIP, PINN (data only), PINN (physics)
MODELS = ["GP", "PCHIP", "PINN-data", "PINN-phys"]
cv = {m: {k: np.zeros(len(SER[k][0])) for k in SER} for m in MODELS}
for f in range(4):
    tr = {}
    for k, (T, y) in SER.items():
        te = np.arange(len(T)) % 4 == f; tr[k] = (T[~te], y[~te])
        cv["GP"][k][te] = pred((fit_stress_gp if k.startswith("sigma") else fit_cal_gp)(T[~te], y[~te], 5), T[te])[0]
        cv["PCHIP"][k][te] = PchipInterpolator(T[~te], y[~te], extrapolate=True)(T[te])
    for name, ph in [("PINN-data", False), ("PINN-phys", True)]:
        nets = [train_pinn(tr, ph, 100 * f + s)[0] for s in range(N_CV)]
        for k, (T, y) in SER.items():
            te = np.arange(len(T)) % 4 == f
            cv[name][k][te] = ser_pred(members(nets, T[te]), T[te], k).mean(0)
    print(f"  CV fold {f + 1}/4 done")
cvt = {m: {k: metr(SER[k][1], cv[m][k]) for k in SER} for m in MODELS}
R["cv4"] = cvt
pd.DataFrame([{"model": m, "series": k, **cvt[m][k]} for m in MODELS for k in SER]).to_csv(f"{OUT}/cv4_table.csv", index=False)

# gap-bridging test (100-200 K stress points withheld)
gm = (T13 < 100) | (T13 > 200); Tgd = np.linspace(20, 450, 431)
R["gap_test_PINN"] = {}
for name, ph in [("PINN-data", False), ("PINN-phys", True)]:
    tr = {k: v for k, v in SER.items()}; tr["sigma_Ms"] = (T13[gm], sMs[gm]); tr["sigma_Af"] = (T13[gm], sAf[gm])
    nets = [train_pinn(tr, ph, 500 + s)[0] for s in range(N_CV)]
    held = ser_pred(members(nets, T13[~gm]), T13[~gm], "sigma_0").mean(0)
    cur = ser_pred(members(nets, Tgd), Tgd, "sigma_0").mean(0)
    R["gap_test_PINN"][name] = {"held_MAE": float(np.mean(np.abs(held - s0[~gm]))), "T_inv_K": first_inversion(cur, Tgd)}

# final ensembles on all data
full = {k: v for k, v in SER.items()}
nets_p, hist_p = [], None
for s in range(N_FIN):
    n_, h_ = train_pinn(full, True, 900 + s, track=(s == 0)); nets_p.append(n_)
    if s == 0: hist_p = np.array(h_)
nets_d = [train_pinn(full, False, 900 + s)[0] for s in range(N_FIN)]
Mp, Md = members(nets_p, Tgd), members(nets_d, Tgd)
sp0 = ser_pred(Mp, Tgd, "sigma_0"); sd0m = ser_pred(Md, Tgd, "sigma_0")
inv_m = np.array([argmin_T(sp0[i], Tgd) for i in range(N_FIN)])
R["inversion_PINN"] = {"mean_curve_K": argmin_T(sp0.mean(0), Tgd), "members_K": inv_m.tolist(),
                       "data_only_mean_curve_K": argmin_T(sd0m.mean(0), Tgd)}
Tz2 = np.linspace(Ts.min(), Ts.max(), 700); Mz = members(nets_p, Tz2)
zcp = np.array([zero_cross(Mz[i][:, 4], Tz2) for i in range(N_FIN)])
R["Tmin_PINN"] = {"mean": float(np.nanmean(zcp)), "sd_ensemble": float(np.nanstd(zcp)), "members": zcp.tolist(), "Song": 97.0}
Mc = members(nets_p, Tcop); Sef, Qq = Mc[..., 4], Mc[..., 2]
copP = np.where(Sef < 0, Tcop[None, :] * np.abs(Sef) / Qq, np.nan)
R["COP_PINN"] = {"at_294_mean": float(np.nanmean(copP[:, i294])), "at_294_sd_ensemble": float(np.nanstd(copP[:, i294])), "Song": 4.6}
# CC violation from the data-only ensemble (physics not imposed)
slope_d = np.gradient(sd0m, Tgd, axis=1); cc_d = -Md[..., 3] * RHO / EPS / 1e6
ex_d = slope_d - cc_d
R["cc_violation_PINN_data_only"] = {
    "mean_excess_50_150": float(ex_d[:, (Tgd >= 50) & (Tgd <= 150)].mean()),
    "mean_excess_250_400": float(ex_d[:, (Tgd >= 250) & (Tgd <= 400)].mean())}

# M3: SYMBOLIC REGRESSION
print("M3: symbolic regression")
py = None
try:
    import sympy as sp
    from pysr import PySRRegressor
    ym, ys_ = s0.mean(), s0.std(); X = (T13 / 100.).reshape(-1, 1); z = (s0 - ym) / ys_
    pk = f"{OUT}/pysr_sigma0_v2.pkl"
    if os.path.exists(pk):
        model = pickle.load(open(pk, "rb"))
    else:
        kw = dict(binary_operators=["+", "-", "*"], unary_operators=["exp", "square", "tanh"], maxsize=12,
                  parsimony=0.001, niterations=PY_IT, populations=20, population_size=40,
                  elementwise_loss="loss(prediction, target) = (prediction - target)^2",
                  verbosity=0, progress=False, temp_equation_file=True)
        try:
            model = PySRRegressor(random_state=SEED, parallelism="serial", deterministic=True, **kw).fit(X, z)
        except Exception:
            model = PySRRegressor(**kw).fit(X, z)
        pickle.dump(model, open(pk, "wb"))
    x0 = sp.Symbol("x0")
    def build(expr):
        cs = sorted(expr.atoms(sp.Float), key=float); sy_ = sp.symbols(f"c0:{len(cs)}") if cs else ()
        f = sp.lambdify((x0, *sy_), expr.xreplace(dict(zip(cs, sy_))), "numpy")
        def pr(xx, p):
            with np.errstate(all="ignore"):
                return np.broadcast_to(np.asarray(f(xx, *p), float), np.shape(xx))
        return pr, np.array([float(c) for c in cs])
    def fitc(pr, p0, xx, zz):
        if not len(p0): return p0
        r = lambda q: np.nan_to_num(pr(xx, q) - zz, nan=1e6, posinf=1e6, neginf=-1e6)
        return least_squares(r, p0, max_nfev=300).x
    eqs = model.equations_.reset_index(drop=True); stats = []
    for i in range(len(eqs)):
        pr, p0 = build(model.sympy(i)); e = np.zeros(13)
        for j in range(13):
            m = np.ones(13, bool); m[j] = False
            e[j] = (pr((T13[j:j + 1] / 100.), fitc(pr, p0, T13[m] / 100., z[m]))[0] - z[j]) * ys_
        stats.append((int(eqs.loc[i, "complexity"]), float(np.mean(e ** 2)), float(np.std(e ** 2) / np.sqrt(13)), e))
    msq = np.array([s[1] for s in stats]); b = int(np.argmin(msq)); thr = msq[b] + stats[b][2]
    cand = [i for i in range(len(stats)) if msq[i] <= thr]; sel_i = min(cand, key=lambda i: stats[i][0])
    expr = model.sympy(sel_i); pr, p0 = build(expr); pf = fitc(pr, p0, T13 / 100., z)
    curve = lambda p, T: pr(np.asarray(T) / 100., p) * ys_ + ym
    cp_ = curve(pf, Tg); e_sel = stats[sel_i][3]
    rng = np.random.RandomState(SEED); boots, invb = [], []
    for _ in range(200):
        ix = rng.choice(13, 13, replace=True)
        if len(np.unique(T13[ix])) < 4: continue
        try:
            pb = fitc(pr, pf, T13[ix] / 100., z[ix]); cb = curve(pb, Tg)
            if np.all(np.isfinite(cb)): boots.append(cb); invb.append(argmin_T(cb, Tg))
        except Exception: pass
    boots = np.array(boots); invb = np.array([v for v in invb if not np.isnan(v)])
    py = dict(eqs=eqs, stats=stats, sel=sel_i, curve=cp_, boots=boots)
    R["pysr"] = {"equation_scaled_x0_is_T_over_100": str(expr), "complexity": stats[sel_i][0],
                 "LOO_RMSE_MPa": float(np.sqrt(stats[sel_i][1])), "LOO_R2": float(1 - np.sum(e_sel ** 2) / np.sum((s0 - s0.mean()) ** 2)),
                 "LOO_MAE_MPa": float(np.mean(np.abs(e_sel))), "inversion_curve_K": argmin_T(cp_, Tg),
                 "inversion_bootstrap_median_K": float(np.median(invb)) if invb.size else None,
                 "inversion_bootstrap_CI95_K": [float(np.percentile(invb, 2.5)), float(np.percentile(invb, 97.5))] if invb.size else None,
                 "selection": "simplest equation within 1 SE of best LOO error; structure search uses all data (not nested)",
                 "best_LOO_complexity": stats[b][0]}
except Exception as e:
    R["pysr"] = {"status": f"skipped: {type(e).__name__}: {str(e)[:150]}"}
    print("  PySR skipped:", R["pysr"]["status"])

# FIGURES
print("Figures")
inv_ci = R["inversion_GP"]["posterior_CI95_K"]

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
a = ax[0]; zones(a); band(a, Tg, mean0 - 1.96 * sd0, mean0 + 1.96 * sd0, PAL["gp"], "GP 95% interval")
a.plot(Tg, mean0, c=PAL["gp"], lw=2, label="GP posterior mean")
a.scatter(T13, sMs, marker="^", s=16, c="#9aa5b1", label="σ$_{Ms}$"); a.scatter(T13, sAf, marker="v", s=16, c="#9aa5b1", label="σ$_{Af}$")
a.scatter(T13, s0, c=PAL["dat"], s=30, zorder=5, label="σ$_0$ (digitised)")
a.axvspan(*inv_ci, color=PAL["inv"], alpha=.15, label="Inversion 95% interval")
a.set(xlabel="T (K)", ylabel="Transformation stress (MPa)", title="(a) Stress-temperature response"); a.legend(); a.set_xlim(0, 460)
a = ax[1]; zones(a); band(a, Tg, np.percentile(dS0, 2.5, 1), np.percentile(dS0, 97.5, 1), PAL["gp"], "GP slope 95%")
a.plot(Tg, d_mean, c=PAL["gp"], lw=2); a.axhline(0, c="k", lw=.8)
a.scatter(Tmid, fd, c=PAL["dat"], s=22, zorder=5, label="Finite difference of data")
a.set(xlabel="T (K)", ylabel="dσ$_0$/dT (MPa/K)", title="(b) Slope inversion"); a.legend(); a.set_xlim(0, 460)
a = ax[2]; zones(a)
band(a, Tv, np.percentile(CCV, 2.5, 1), np.percentile(CCV, 97.5, 1), "#f4a261", "CC-implied slope 95%")
a.plot(Tv, CCV.mean(1), c="#e76f51", lw=2, label="CC-implied: -ΔS ρ/ε$_{tr}$")
band(a, Tv, np.percentile(SV, 2.5, 1), np.percentile(SV, 97.5, 1), PAL["gp"], "Measured slope 95%")
a.plot(Tv, SV.mean(1), c=PAL["gp"], lw=2, label="Measured dσ$_0$/dT"); a.axhline(0, c="k", lw=.8)
a.set(xlabel="T (K)", ylabel="Slope (MPa/K)", title="(c) Clausius-Clapeyron violation"); a.legend(); a.set_xlim(40, 410)
plt.tight_layout(); plt.savefig(f"{OUT}/fig1_inversion_and_CC.png", dpi=220); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
a = ax[0]; lim = [s0.min() - 20, s0.max() + 20]; a.plot(lim, lim, "k--", lw=.8)
a.scatter(s0, R["loo_pred_sigma_0_GP"], c=T13, cmap="viridis", s=45, edgecolor="w", zorder=5)
a.set(xlabel="Measured σ$_0$ (MPa)", ylabel="LOO prediction (MPa)",
      title=f"(a) GP leave-one-out, R²={stress_loo['sigma_0']['GP']['R2']:.3f}")
a = ax[1]; zones(a); g = fit_stress_gp(T13[gm], s0[gm], 10); mg, sg = pred(g, Tg)
band(a, Tg, mg - 1.96 * sg, mg + 1.96 * sg, PAL["gp"]); a.plot(Tg, mg, c=PAL["gp"], lw=2, label="GP, 100-200 K withheld")
a.scatter(T13[gm], s0[gm], c=PAL["dat"], s=30, label="Used"); a.scatter(T13[~gm], s0[~gm], marker="*", s=140, c=PAL["pinn"], zorder=6, label="Withheld")
a.set(xlabel="T (K)", ylabel="σ$_0$ (MPa)", title=f"(b) Gap bridging, MAE={R['gap_test_GP']['held_MAE']:.1f} MPa"); a.legend(); a.set_xlim(0, 460)
a = ax[2]; ks = ["GP", "PCHIP", "poly1", "poly2", "poly3"]
a.bar(ks, [stress_loo["sigma_0"][k]["MAE"] for k in ks], color=[PAL["gp"], PAL["pc"]] + ["#cfd4da"] * 3)
a.set(ylabel="LOO MAE on σ$_0$ (MPa)", title="(c) Baseline comparison"); a.set_yscale("log")
plt.tight_layout(); plt.savefig(f"{OUT}/fig2_validation.png", dpi=220); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
a = ax[0]; zones(a); mS, sS = pred(GS, Tg)
band(a, Tg, mS - 1.96 * sS, mS + 1.96 * sS, PAL["gp"], "GP 95%"); a.plot(Tg, mS, c=PAL["gp"], lw=2, label="GP")
a.plot(Tgd, Mp[..., 4].mean(0), c=PAL["pinn"], lw=2, ls="--", label="PINN ensemble mean")
a.scatter(Ts, S, c=PAL["dat"], s=28, zorder=5, label="Data"); a.axhline(0, c="k", lw=.8)
a.axvline(R["Tmin_GP"]["Song"], c="#f4a261", lw=2, label="Song 2026: 97 K")
a.set(xlabel="T (K)", ylabel="ΔS$_{eff}$ (J/kg K)", title="(a) Effective entropy zero crossing"); a.legend(); a.set_xlim(0, 450)
a = ax[1]; a.hist(zc, bins=30, color=PAL["gp"], alpha=.6, label="GP posterior")
for v in zcp[~np.isnan(zcp)]: a.axvline(v, c=PAL["pinn"], lw=1.5, alpha=.8)
a.axvline(97, c="#f4a261", lw=2.5, label="Song 2026"); a.plot([], [], c=PAL["pinn"], label="PINN members")
a.set(xlabel="T$_{min}$ (K)", ylabel="Count", title="(b) Cooling onset temperature"); a.legend()
a = ax[2]; band(a, Tcop, cop_lo, cop_hi, PAL["gp"], "GP joint 95%"); a.plot(Tcop, cop_med, c=PAL["gp"], lw=2, label="GP median")
a.plot(Tcop, np.nanmean(copP, 0), c=PAL["pinn"], lw=2, ls="--", label="PINN ensemble mean")
a.scatter([294], [4.6], marker="*", s=220, c="gold", edgecolor="k", zorder=6, label="Song 2026: 4.6")
a.set(xlabel="T (K)", ylabel="COP$_{mat}$", title="(c) Material COP, measured range only"); a.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig3_caloric.png", dpi=220); plt.close()

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
a = ax[0, 0]; a.semilogy(hist_p[:, 0], c="k", label="Total"); a.semilogy(hist_p[:, 1], c=PAL["pinn"], label="Data")
a.semilogy(hist_p[:, 2] + 1e-9, c=PAL["gp"], label="Physics (CC + Song)"); a.set(xlabel="Epoch", title="(a) Training"); a.legend()
a = ax[0, 1]; zones(a)
for M_, c_, l_ in [(sp0, PAL["pinn"], "PINN + physics"), (sd0m, PAL["pysr"], "PINN data only")]:
    band(a, Tgd, M_.mean(0) - 1.96 * M_.std(0), M_.mean(0) + 1.96 * M_.std(0), c_); a.plot(Tgd, M_.mean(0), c=c_, lw=2, label=l_)
a.scatter(T13, s0, c=PAL["dat"], s=28, zorder=5, label="Data"); a.set(xlabel="T (K)", ylabel="σ$_0$ (MPa)", title="(b) Stress (ensemble ±1.96 sd)"); a.legend(); a.set_xlim(0, 460)
a = ax[0, 2]; zones(a); a.plot(Tgd, slope_d.mean(0), c=PAL["pysr"], lw=2, label="Learned slope (data-only PINN)")
a.plot(Tgd, cc_d.mean(0), c="#e76f51", lw=2, ls="--", label="CC-implied slope"); a.axhline(0, c="k", lw=.8)
a.set(xlabel="T (K)", ylabel="MPa/K", title="(c) CC residual map"); a.legend(); a.set_xlim(40, 410)
a = ax[1, 0]; zones(a); a.plot(Tgd, Mp[..., 2].mean(0), c=PAL["pinn"], lw=2, label="PINN"); band(a, Tgd, Mp[..., 2].mean(0) - 1.96 * Mp[..., 2].std(0), Mp[..., 2].mean(0) + 1.96 * Mp[..., 2].std(0), PAL["pinn"])
a.scatter(Tq, Q, c=PAL["dat"], s=28, label="Data"); a.set(xlabel="T (K)", ylabel="Q$_{diss}$ (J/kg)", title="(d) Dissipation"); a.legend(); a.set_xlim(0, 460)
a = ax[1, 1]; zones(a); dpred = ser_pred(Mp, Tgd, "dTad")
a.plot(Tgd, dpred.mean(0), c=PAL["pinn"], lw=2, label="PINN T·ΔS$_{eff}$/C$_p$"); band(a, Tgd, dpred.mean(0) - 1.96 * dpred.std(0), dpred.mean(0) + 1.96 * dpred.std(0), PAL["pinn"])
a.scatter(Tc, dTm, c=PAL["dat"], s=28, zorder=5, label="Measured ΔT$_{ad}$"); a.axhline(0, c="k", lw=.8)
a.set(xlabel="T (K)", ylabel="ΔT$_{ad}$ (K)", title="(e) Adiabatic temperature change"); a.legend(); a.set_xlim(0, 460)
a = ax[1, 2]; sers = list(SER); w = .2
for i, m in enumerate(MODELS):
    a.bar(np.arange(len(sers)) + (i - 1.5) * w, [np.clip(cvt[m][k]["R2"], -.5, 1) for k in sers], w, label=m,
          color=[PAL["gp"], PAL["pc"], PAL["pysr"], PAL["pinn"]][i])
a.set_xticks(range(len(sers))); a.set_xticklabels(sers, rotation=30, ha="right"); a.set(ylabel="4-fold CV R² (clipped)", title="(f) Cross-validation"); a.legend(ncol=2)
plt.tight_layout(); plt.savefig(f"{OUT}/fig4_pinn.png", dpi=220); plt.close()

if py is not None:
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6)); st = py["stats"]
    a = ax[0]; a.plot([s[0] for s in st], [np.sqrt(s[1]) for s in st], "o-", c=PAL["pysr"])
    a.scatter([st[py["sel"]][0]], [np.sqrt(st[py["sel"]][1])], marker="*", s=240, c="gold", edgecolor="k", zorder=5, label="Selected (1-SE rule)")
    a.set(xlabel="Complexity", ylabel="LOO RMSE (MPa)", title="(a) Pareto front"); a.set_yscale("log"); a.legend()
    a = ax[1]; zones(a); b = py["boots"]
    if len(b): band(a, Tg, np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0), PAL["pysr"], "Bootstrap 95%")
    a.plot(Tg, py["curve"], c=PAL["pysr"], lw=2, label="Selected equation"); a.scatter(T13, s0, c=PAL["dat"], s=28, zorder=5, label="Data")
    a.set(xlabel="T (K)", ylabel="σ$_0$ (MPa)", title="(b) Symbolic fit"); a.legend(); a.set_xlim(0, 460)
    a = ax[2]; zones(a); a.plot(Tg, np.gradient(py["curve"], Tg), c=PAL["pysr"], lw=2, label="Equation slope")
    a.plot(Tg, d_mean, c=PAL["gp"], lw=2, ls="--", label="GP slope"); a.axhline(0, c="k", lw=.8)
    a.set(xlabel="T (K)", ylabel="dσ$_0$/dT (MPa/K)", title="(c) Slope comparison"); a.legend(); a.set_xlim(0, 460)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig5_pysr.png", dpi=220); plt.close()

fig, a = plt.subplots(figsize=(9, 3.8)); Mx = np.array([[cvt[m][k]["R2"] for k in sers] for m in MODELS])
im = a.imshow(np.clip(Mx, 0, 1), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
for i in range(len(MODELS)):
    for j in range(len(sers)): a.text(j, i, f"{Mx[i, j]:.2f}", ha="center", va="center", fontsize=9)
a.set_xticks(range(len(sers))); a.set_xticklabels(sers, rotation=30, ha="right"); a.set_yticks(range(len(MODELS))); a.set_yticklabels(MODELS)
a.grid(False); a.set_title("4-fold interleaved cross-validation R²"); plt.colorbar(im, ax=a)
plt.tight_layout(); plt.savefig(f"{OUT}/fig6_cv_heatmap.png", dpi=220); plt.close()


# EXTENSION: multi-alloy analysis
print("Extension: multi-alloy analysis")
import re
from scipy.stats import t as _tdist
XA = {}

def _ser(alloy, prop):
    s = df[(df.alloy_system == alloy) & (df.property == prop)].dropna(subset=["T_K", "value"]).sort_values("T_K")
    return s["T_K"].values.astype(float), s["value"].values.astype(float)

def _ols(T, y):
    p = np.polyfit(T, y, 1); n = len(T)
    if n < 3:
        return float(p[0]), np.nan, np.nan
    r = y - np.polyval(p, T)
    se = np.sqrt((r @ r) / (n - 2) / ((T - T.mean()) ** 2).sum())
    h = _tdist.ppf(.975, n - 2) * se
    return float(p[0]), float(p[0] - h), float(p[0] + h)

# 1. per-alloy stress-temperature table (all dataset points, used as given)
tn, msv = _ser("Ti-Ni-Fe", "sigma_Ms"); tn2, afv = _ser("Ti-Ni-Fe", "sigma_Af")
assert np.array_equal(tn, tn2)
XS = [("Ti-Al-Cr", "sigma_0", (T13, s0)), ("Ti-Ni-Fe", "sigma_0 (midpoint)", (tn, .5 * (msv + afv))),
      ("Ti-Ni-Fe", "sigma_Ms", (tn, msv)), ("Ti-Ni-Fe", "sigma_Af", (tn, afv)),
      ("Ti-Ni-Fe", "upper_plateau", _ser("Ti-Ni-Fe", "upper_plateau")),
      ("Ti-Ni-Fe", "lower_plateau", _ser("Ti-Ni-Fe", "lower_plateau")),
      ("Mg-Sc", "sigma_c", _ser("Mg-Sc", "sigma_c")), ("Ti-Ni-Co", "sigma_c", _ser("Ti-Ni-Co", "sigma_c")),
      ("Ti-Nb", "sigma_c", _ser("Ti-Nb", "sigma_c")), ("Cu-Al-Mn", "sigma_c", _ser("Cu-Al-Mn", "sigma_c")),
      ("Fe-Mn-Al-Ni", "sigma_c", _ser("Fe-Mn-Al-Ni", "sigma_c"))]
xa_rows = []
for al, pr, (Tx, yx) in XS:
    sl, lo_, hi_ = _ols(Tx, yx)
    row = {"alloy": al, "series": pr, "n": len(Tx), "T_min_K": float(Tx.min()), "T_max_K": float(Tx.max()),
           "slope_all": sl, "slope_lo95": lo_, "slope_hi95": hi_, "T_star_K": np.nan, "sigma_star": np.nan,
           "slope_below": np.nan, "slope_above": np.nan, "depth": np.nan}
    i = int(np.argmin(yx))
    if 0 < i < len(Tx) - 1:
        row.update(T_star_K=float(Tx[i]), sigma_star=float(yx[i]), slope_below=_ols(Tx[:i + 1], yx[:i + 1])[0],
                   slope_above=_ols(Tx[i:], yx[i:])[0], depth=float((yx[0] - yx[i]) / yx[i]))
    xa_rows.append(row)
xa_tab = pd.DataFrame(xa_rows); xa_tab.to_csv(f"{OUT}/multi_alloy_table.csv", index=False)
XA["alloy_table"] = xa_rows

# 2. Cu-Al-Mn transformation temperatures by processing route
cm = df[(df.alloy_system == "Cu-Al-Mn") & df.property.isin(["Ms", "Mf", "As", "Af"])]
pv = cm.pivot_table(index="processing", columns="property", values="value", aggfunc="first").dropna().sort_values("Ms")
pv["T0"] = (pv.Ms + pv.Af) / 2
pv["hysteresis"] = (pv.As + pv.Af - pv.Ms - pv.Mf) / 2
pv["span_M"] = pv.Ms - pv.Mf; pv["span_A"] = pv.Af - pv.As
pv["anneal_K"] = pv.index.str.extract(r"_(\d{3})(?:\s|$)")[0].astype(float).values
pv.to_csv(f"{OUT}/cu_al_mn_table.csv")
comp = cm["composition"].iloc[0]
al_w = float(re.search(r"([\d.]+)Al", comp).group(1)); mn_w = float(re.search(r"([\d.]+)Mn", comp).group(1))
ms_reg = 1192 - 25.5 * al_w - 73.2 * mn_w
an = pv.dropna(subset=["anneal_K"])
XA["cu_al_mn"] = {"table": pv.to_dict("index"), "composition": comp, "Ms_regression_quoted_K": float(ms_reg),
                  "Ms_measured_range_K": [float(pv.Ms.min()), float(pv.Ms.max())],
                  "Ms_shift_per_K_anneal": float(np.polyfit(an.anneal_K, an.Ms, 1)[0]) if len(an) >= 2 else None,
                  "Af_shift_per_K_anneal": float(np.polyfit(an.anneal_K, an.Af, 1)[0]) if len(an) >= 2 else None,
                  "n_anneal_points": int(len(an)), "all_Af_below_150K": bool((pv.Af < 150).all())}

# 3. dissipation and modulus trends, anchored at 273 K
Te, Ev = _ser("Ti-Ni-Fe", "E_modulus"); Th, Hv = _ser("Ti-Ni-Fe", "hysteresis")
Hn = Hv / np.interp(273., Th, Hv); Qn = Q / np.interp(273., Tq, Q); En = Ev / np.interp(273., Te, Ev)
XA["dissipation_trend"] = {"TiNiFe_hysteresis_slope_Jcm3_per_K": _ols(Th, Hv), "TiNiFe_E_slope_GPa_per_K": _ols(Te, Ev),
                           "TiNiFe_hysteresis_ratio_103K_over_273K": float(np.interp(103., Th, Hn)),
                           "TiAlCr_Qdiss_ratio_103K_over_273K": float(np.interp(103., Tq, Qn)),
                           "TiNiFe_E_ratio_103K_over_273K": float(np.interp(103., Te, En))}

# 4. paired comparison of models on the shared 4-fold CV
def _ne(m, k):
    y = SER[k][1]; return np.abs(y - cv[m][k]) / y.std()
grp = {"stress": ["sigma_0", "sigma_Ms", "sigma_Af"], "caloric": ["Qdiss", "DeltaSeff", "DeltaS_cal", "dTad"]}
rngx = np.random.RandomState(7); XA["paired_cv"] = {}
for g_, ks in grp.items():
    for a_, b_ in [("PINN-phys", "PINN-data"), ("GP", "PINN-phys"), ("GP", "PINN-data"), ("GP", "PCHIP")]:
        ea = np.concatenate([_ne(a_, k) for k in ks]); eb = np.concatenate([_ne(b_, k) for k in ks]); d = ea - eb
        bs = np.array([d[rngx.randint(0, len(d), len(d))].mean() for _ in range(2000)])
        try: p_ = float(wilcoxon(ea, eb).pvalue)
        except Exception: p_ = None
        XA["paired_cv"][f"{g_}: {a_} minus {b_}"] = {"n": int(len(d)), "mean_diff": float(d.mean()),
            "CI95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "wilcoxon_p": p_}

# 5. gap-window sweep on sigma_0 (Ti-Al-Cr)
XW = [(60, 120), (130, 200), (200, 280)]; xa_gap = {}
for lo_, hi_ in XW:
    mk = (T13 < lo_) | (T13 > hi_); hd = ~mk; row = {"held_T": T13[hd].tolist()}
    g = fit_stress_gp(T13[mk], s0[mk], 10); pg, sg = pred(g, T13[hd])
    row["GP"] = {"MAE": float(np.mean(np.abs(pg - s0[hd]))), "inside95": int((np.abs(pg - s0[hd]) < 1.96 * sg).sum()),
                 "T_inv": first_inversion(pred(g, Tg)[0], Tg)}
    pp = PchipInterpolator(T13[mk], s0[mk], extrapolate=True)
    row["PCHIP"] = {"MAE": float(np.mean(np.abs(pp(T13[hd]) - s0[hd]))), "T_inv": first_inversion(pp(Tg), Tg)}
    for nm_, ph_ in [("PINN-data", False), ("PINN-phys", True)]:
        tr = dict(SER); tr["sigma_Ms"] = (T13[mk], sMs[mk]); tr["sigma_Af"] = (T13[mk], sAf[mk])
        nn_ = [train_pinn(tr, ph_, 700 + s)[0] for s in range(N_CV)]
        hp = ser_pred(members(nn_, T13[hd]), T13[hd], "sigma_0").mean(0)
        cu = ser_pred(members(nn_, Tgd), Tgd, "sigma_0").mean(0)
        row[nm_] = {"MAE": float(np.mean(np.abs(hp - s0[hd]))), "T_inv": argmin_T(cu, Tgd)}
    xa_gap[f"{lo_}-{hi_} K"] = row
    print(f"  window {lo_}-{hi_} K done")
XA["window_sweep"] = xa_gap
R["multi_alloy"] = XA

# figures
fig, ax = plt.subplots(2, 2, figsize=(14, 10))
a = ax[0, 0]; zones(a); cols = plt.cm.tab10(np.arange(7))
for (al, pr, (Tx, yx)), c_ in zip([x for x in XS if x[1] in ("sigma_0", "sigma_0 (midpoint)", "sigma_c")], cols):
    a.plot(Tx, yx, "o-", c=c_, ms=4, lw=1.5, label=f"{al}")
a.set(xlabel="T (K)", ylabel="Transformation stress (MPa)", title="(a) Stress-temperature, all alloys"); a.legend(ncol=2); a.set_xlim(0, 460)
a = ax[0, 1]; lab = [f"{r['alloy']}\n{r['series']}" for r in xa_rows]; sl_ = xa_tab.slope_all.values
err = np.vstack([np.nan_to_num(sl_ - xa_tab.slope_lo95.values), np.nan_to_num(xa_tab.slope_hi95.values - sl_)])
a.bar(range(len(sl_)), sl_, yerr=err, capsize=3, color=[PAL["gp"] if v > 0 else PAL["pinn"] for v in sl_])
a.axhline(0, c="k", lw=.8); a.set_xticks(range(len(lab))); a.set_xticklabels(lab, rotation=60, ha="right", fontsize=7)
a.set(ylabel="OLS slope (MPa/K), 95% t-interval", title="(b) Slopes over each measured range")
a = ax[1, 0]
for i, (nm, r_) in enumerate(pv.iterrows()):
    a.hlines(i + .15, r_.Mf, r_.Ms, colors=PAL["gp"], lw=7); a.hlines(i - .15, r_.As, r_.Af, colors=PAL["pinn"], lw=7)
    a.plot([r_.T0], [i], "k|", ms=14)
a.plot([], [], c=PAL["gp"], lw=6, label="Mf to Ms"); a.plot([], [], c=PAL["pinn"], lw=6, label="As to Af")
a.set_yticks(range(len(pv))); a.set_yticklabels([n.split()[0] for n in pv.index]); a.axvline(150, c="#f4a261", ls="--", lw=1)
a.set(xlabel="T (K)", title="(c) Cu-Al-Mn transformation intervals (black tick: T0)"); a.legend()
a = ax[1, 1]; zones(a)
a.plot(Th, Hn, "o-", c=PAL["pysr"], label="Ti-Ni-Fe hysteresis"); a.plot(Te, En, "s-", c=PAL["pc"], label="Ti-Ni-Fe E modulus")
a.plot(Tq, Qn, "^-", c=PAL["pinn"], ms=4, label="Ti-Al-Cr Qdiss"); a.axhline(1, c="k", lw=.8)
a.set(xlabel="T (K)", ylabel="Value / value at 273 K", title="(d) Dissipation and modulus, normalised"); a.legend(); a.set_xlim(0, 460)
plt.tight_layout(); plt.savefig(f"{OUT}/fig7_multi_alloy.png", dpi=220); plt.close()

fig, a = plt.subplots(figsize=(9, 4)); mods = ["GP", "PCHIP", "PINN-data", "PINN-phys"]
for i, m_ in enumerate(mods):
    a.bar(np.arange(len(XW)) + (i - 1.5) * .2, [xa_gap[k][m_]["MAE"] for k in xa_gap], .2, label=m_,
          color=[PAL["gp"], PAL["pc"], PAL["pysr"], PAL["pinn"]][i])
a.set_xticks(range(len(XW))); a.set_xticklabels(list(xa_gap)); a.set(ylabel="Held-out MAE on σ$_0$ (MPa)", title="Gap-window sweep (Ti-Al-Cr)"); a.legend(ncol=2)
plt.tight_layout(); plt.savefig(f"{OUT}/fig8_window_sweep.png", dpi=220); plt.close()

# claims text
XL = ["MULTI-ALLOY EXTENSION", "-" * 70]
for r in xa_rows:
    XL.append(f"{r['alloy']:12s} {r['series']:20s} n={r['n']} {r['T_min_K']:.0f}-{r['T_max_K']:.0f} K slope={r['slope_all']:+.3f} "
              f"[{r['slope_lo95']:+.3f},{r['slope_hi95']:+.3f}] T*={r['T_star_K']:.0f} below={r['slope_below']:+.2f} above={r['slope_above']:+.2f} depth={r['depth']:.3f}")
XL += ["", "Cu-Al-Mn by processing route:", pv.round(1).to_string(),
       f"Quoted Ms regression for {comp}: {ms_reg:.1f} K vs measured {pv.Ms.min():.0f}-{pv.Ms.max():.0f} K",
       f"Anneal shift (two points): Ms {XA['cu_al_mn']['Ms_shift_per_K_anneal']} K/K, Af {XA['cu_al_mn']['Af_shift_per_K_anneal']} K/K",
       "", f"Dissipation trend: {XA['dissipation_trend']}", "", "Paired CV comparison (negative = first model better):"]
for k_, v_ in XA["paired_cv"].items():
    XL.append(f"  {k_}: n={v_['n']} diff={v_['mean_diff']:+.3f} CI={[round(x, 3) for x in v_['CI95']]} p={v_['wilcoxon_p']}")
XL += ["", "Gap-window sweep (held-out MAE on sigma_0, MPa):"]
for k_, v_ in xa_gap.items():
    XL.append(f"  {k_} held {v_['held_T']}: " + " | ".join(f"{m_} {v_[m_]['MAE']:.1f} (inv {v_[m_]['T_inv']})" for m_ in mods))
XL.append("")


# EXTENSION 2: validation and cryogenic metrics
print("Extension 2: validation and cryogenic metrics")
from scipy.integrate import cumulative_trapezoid as _ctz
XB = {}

# 1. CC violation robustness to the open (rho, eps_tr) pairs and to a data-calibrated scale
dSd = draws(GSC, Tv, seed=12)
def _wn(ex, lo, hi):
    w = (Tv >= lo) & (Tv <= hi); x = ex[w].mean(0)
    return {"mean": float(x.mean()), "CI95": [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))],
            "P_negative": float((x < 0).mean())}
sens = {}
for rho_, eps_ in [(4500., .064), (4360., .073)]:
    cc = -dSd * rho_ / eps_ / 1e6
    sens[f"rho{rho_:.0f}_eps{eps_}"] = {"50_150": _wn(SV - cc, 50, 150), "250_400": _wn(SV - cc, 250, 400)}
cc0 = -dSd * RHO / EPS / 1e6; mh = (Tv >= 250) & (Tv <= 400)
kcal = float(SV[mh].mean() / cc0[mh].mean())
sens["rescaled_to_250_400"] = {"scale": kcal, "50_150": _wn(SV - kcal * cc0, 50, 150)}
XB["cc_sensitivity"] = sens

# 2. CC closure: integrate calorimetric dS from the 294 K anchor and compare with measured sigma_0
ia = int(np.argmin(np.abs(Tv - 294.)))
integ = _ctz(CCV.mean(1), Tv, initial=0.)
sig_gp = np.interp(Tv, Tg, mean0); sig_cc = sig_gp[ia] + integ - integ[ia]; dfc = sig_gp - sig_cc
nm_ = np.arange(len(Tv)) != ia; i150 = int(np.argmin(np.abs(Tv - 150.)))
XB["cc_closure"] = {"anchor_T_K": float(Tv[ia]), "RMSE_MPa": float(np.sqrt(np.mean(dfc[nm_] ** 2))),
                    "deficit_50K_MPa": float(dfc[0]), "deficit_150K_MPa": float(dfc[i150]),
                    "deficit_50K_Jkg": float(dfc[0] * EPS * 1e6 / RHO),
                    "max_abs_250_400_MPa": float(np.abs(dfc[mh]).max())}

# 3. Carnot-normalised efficiency and heat lift
GD = fit_cal_gp(Tc, dTm, 20)
Sx = draws(GS, Tcop, seed=21); Dx = draws(GD, Tcop, seed=22); Qx = draws(GQ, Tcop, seed=23)
ETA = np.where(Sx < 0, np.abs(Sx) * np.abs(Dx) / Qx, np.nan); QCL = Tcop[:, None] * np.abs(Sx)
carn = []
for t_ in [120, 150, 200, 250, 294, 350, 400]:
    i_ = int(np.argmin(np.abs(Tcop - t_)))
    carn.append({"T_K": float(Tcop[i_]), "eta_median": float(np.nanmedian(ETA[i_])),
                 "eta_CI95": [float(np.nanpercentile(ETA[i_], 2.5)), float(np.nanpercentile(ETA[i_], 97.5))],
                 "heat_lift_Jkg": float(np.nanmedian(QCL[i_])), "heat_lift_Jcm3": float(np.nanmedian(QCL[i_]) * RHO / 1e6)})
XB["carnot_efficiency"] = carn

# 4. conformal calibration of GP intervals (leave-one-out z-scores)
def _cf(T, y, fitter):
    o, s = [], []
    for i in range(len(T)):
        m = np.ones(len(T), bool); m[i] = False
        a, b = pred(fitter(T[m], y[m], 5), T[i]); o.append(a[0]); s.append(b[0])
    z = np.abs(y - np.array(o)) / np.array(s); n = len(z)
    q = float(np.quantile(z, min(1., np.ceil((n + 1) * .95) / n)))
    return {"n": n, "PICP_at_1.96": float(np.mean(z < 1.96)), "calibrated_multiplier": q, "inflation_factor": q / 1.96}
XB["interval_calibration"] = {"sigma_0": _cf(T13, s0, fit_stress_gp), "DeltaSeff": _cf(Ts, S, fit_cal_gp),
                              "dTad": _cf(Tc, dTm, fit_cal_gp)}

# 5. digitisation-noise propagation (T +/- 3 K on all series, stress +/- 2 MPa; DeltaSeff values unperturbed)
rs = np.random.RandomState(11); inv_d, tm_d = [], []
for _ in range(15 if FAST else 40):
    Tj = T13 + rs.normal(0, 3, 13); o = np.argsort(Tj)
    g = fit_stress_gp(Tj[o], (s0 + rs.normal(0, 2, 13))[o], 3); inv_d.append(argmin_T(pred(g, Tg)[0], Tg))
    Tj = Ts + rs.normal(0, 3, len(Ts)); o = np.argsort(Tj)
    g = fit_cal_gp(Tj[o], S[o], 3); tm_d.append(zero_cross(pred(g, Tz)[0], Tz))
inv_d = np.array([v for v in inv_d if not np.isnan(v)]); tm_d = np.array([v for v in tm_d if not np.isnan(v)])
XB["digitisation_noise"] = {"n": int(len(inv_d)), "inversion_mean_K": float(inv_d.mean()), "inversion_sd_K": float(inv_d.std(ddof=1)),
                            "inversion_CI95_K": [float(np.percentile(inv_d, 2.5)), float(np.percentile(inv_d, 97.5))],
                            "Tmin_mean_K": float(tm_d.mean()), "Tmin_sd_K": float(tm_d.std(ddof=1)),
                            "Tmin_CI95_K": [float(np.percentile(tm_d, 2.5)), float(np.percentile(tm_d, 97.5))]}

# 6. Holm correction and per-series win counts for the paired model comparison
pc = R["multi_alloy"]["paired_cv"]; ks = [k for k in pc if pc[k]["wilcoxon_p"] is not None]
ps = np.array([pc[k]["wilcoxon_p"] for k in ks]); od = np.argsort(ps); adj = np.zeros(len(ps)); run = 0.
for r_, i_ in enumerate(od):
    run = max(run, (len(ps) - r_) * ps[i_]); adj[i_] = min(1., run)
XB["holm_adjusted_p"] = {k: float(a) for k, a in zip(ks, adj)}
wins = {}
for g_, kk in grp.items():
    for a_, b_ in [("PINN-phys", "PINN-data"), ("GP", "PINN-phys"), ("GP", "PINN-data"), ("GP", "PCHIP")]:
        wins[f"{g_}: {a_} lower error than {b_}"] = f"{sum(_ne(a_, k).mean() < _ne(b_, k).mean() for k in kk)} of {len(kk)} series"
XB["series_wins"] = wins
R["extra"] = XB

# figure
fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))
a = ax[0, 0]; nm = list(sens); md = [sens[k]["50_150"]["mean"] for k in nm]
er = np.array([[sens[k]["50_150"]["mean"] - sens[k]["50_150"]["CI95"][0], sens[k]["50_150"]["CI95"][1] - sens[k]["50_150"]["mean"]] for k in nm]).T
a.bar(range(len(nm)), md, yerr=er, capsize=4, color=[PAL["gp"], PAL["pc"], PAL["pysr"]]); a.axhline(0, c="k", lw=.8)
a.set_xticks(range(len(nm))); a.set_xticklabels([k.replace("_", "\n") for k in nm], fontsize=8)
a.set(ylabel="Excess slope 50-150 K (MPa/K)", title="(a) CC violation under parameter alternatives")
a = ax[0, 1]; zones(a); a.plot(Tv, sig_gp, c=PAL["gp"], lw=2, label="Measured (GP mean)")
a.plot(Tv, sig_cc, c="#e76f51", lw=2, ls="--", label="Integrated calorimetric CC"); a.scatter(T13, s0, c=PAL["dat"], s=22, zorder=5)
a.set(xlabel="T (K)", ylabel="σ$_0$ (MPa)", title=f"(b) CC closure, RMSE={XB['cc_closure']['RMSE_MPa']:.0f} MPa"); a.legend(); a.set_xlim(40, 410)
a = ax[1, 0]; zones(a); em = np.nanmedian(ETA, 1)
band(a, Tcop, np.nanpercentile(ETA, 2.5, 1), np.nanpercentile(ETA, 97.5, 1), PAL["gp"], "Joint 95%")
a.plot(Tcop, em, c=PAL["gp"], lw=2, label="Median")
a.set(xlabel="T (K)", ylabel="η = |ΔS$_{eff}$||ΔT$_{ad}$|/Q$_{diss}$", title="(c) Efficiency relative to Carnot"); a.legend(); a.set_xlim(100, 410)
a = ax[1, 1]; a.hist(inv_d, bins=12, color=PAL["inv"], alpha=.7, label="Inversion T")
a.hist(tm_d, bins=12, color=PAL["gp"], alpha=.7, label="T$_{min}$"); a.axvline(150, c="#f4a261", lw=2); a.axvline(97, c="#f4a261", lw=2, ls="--")
a.set(xlabel="T (K)", ylabel="Count", title="(d) Digitisation-noise propagation"); a.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig9_validation.png", dpi=220); plt.close()

# claims text
XL += ["", "EXTENSION 2: VALIDATION AND CRYOGENIC METRICS", "-" * 70, "CC violation sensitivity (excess = measured minus CC-implied slope, MPa/K):"]
for k_, v_ in sens.items():
    w_ = v_["50_150"]; XL.append(f"  {k_}: 50-150 K mean {w_['mean']:+.2f} CI {[round(x, 2) for x in w_['CI95']]} P(neg)={w_['P_negative']:.3f}"
                                  + (f" | 250-400 K mean {v_['250_400']['mean']:+.2f}" if "250_400" in v_ else f" | scale {v_['scale']:.2f}"))
XL.append(f"CC closure: {XB['cc_closure']}")
XL.append("Carnot-normalised efficiency and heat lift:")
for r_ in carn: XL.append(f"  {r_['T_K']:.0f} K: eta {r_['eta_median']:.3f} {[round(x, 3) for x in r_['eta_CI95']]}  heat lift {r_['heat_lift_Jkg']:.0f} J/kg ({r_['heat_lift_Jcm3']:.2f} J/cm3)")
XL.append(f"Interval calibration: {XB['interval_calibration']}")
XL.append(f"Digitisation noise: {XB['digitisation_noise']}")
XL.append("Holm-adjusted p: " + "; ".join(f"{k_}: {v_:.4f}" for k_, v_ in XB["holm_adjusted_p"].items()))
XL.append("Series wins: " + "; ".join(f"{k_}: {v_}" for k_, v_ in wins.items()))
XL.append("")
print("Extension 3: reviewer-proofing")
from sklearn.model_selection import KFold
X3 = {}
NREP = 3 if FAST else 10; NSIM = 200 if FAST else 1000; NBOOT3 = 100 if FAST else 500

# A. GP with noise fixed from digitisation uncertainty (2 MPa, 3 K propagated through local slope)
def fit_noise_gp(T, y, restarts=3, sT=3., sY=2.):
    sx, sy = StandardScaler(), StandardScaler()
    Xn = sx.fit_transform(T.reshape(-1, 1)); zn = sy.fit_transform(y.reshape(-1, 1)).ravel()
    var = (sY ** 2 + (np.gradient(y, T) * sT) ** 2) / sy.scale_[0] ** 2
    k = (C(1., (1e-2, 1e2)) * DotProduct(sigma_0=1., sigma_0_bounds=(1e-2, 1e2))
         + C(.5, (1e-2, 1e2)) * Matern(1., (.2, 10.), nu=2.5))
    return GPR(k, alpha=var + 1e-8, n_restarts_optimizer=restarts, random_state=SEED).fit(Xn, zn), sx, sy

def loo_noise(T, y):
    o, s = [], []
    for i in range(len(T)):
        m3 = np.ones(len(T), bool); m3[i] = False
        a3, b3 = pred(fit_noise_gp(T[m3], y[m3]), T[i])
        sl = np.interp(T[i], T[m3], np.gradient(y[m3], T[m3]))
        o.append(a3[0]); s.append(np.sqrt(b3[0] ** 2 + 4. + (sl * 3.) ** 2))
    o, s = np.array(o), np.array(s)
    return {**metr(y, o), "PICP95": float(np.mean(np.abs(y - o) / s < 1.96)), "mean_pred_sd": float(s.mean())}
X3["noise_aware_gp_loo"] = {n_: loo_noise(T13, y_) for n_, y_ in [("sigma_0", s0), ("sigma_Ms", sMs), ("sigma_Af", sAf)]}

# B. repeated random 4-fold CV, GP vs PCHIP, all series
rep = {}
for k_, (T_, y_) in SER.items():
    fitter = fit_stress_gp if k_.startswith("sigma") else fit_cal_gp
    dgs, nwin = [], 0
    for r_ in range(NREP):
        pg3, pp3 = np.zeros(len(T_)), np.zeros(len(T_))
        for tr_, te_ in KFold(4, shuffle=True, random_state=r_).split(T_):
            pg3[te_] = pred(fitter(T_[tr_], y_[tr_], 2), T_[te_])[0]
            pp3[te_] = PchipInterpolator(T_[tr_], y_[tr_], extrapolate=True)(T_[te_])
        eg, ep = np.mean(np.abs(y_ - pg3)) / y_.std(), np.mean(np.abs(y_ - pp3)) / y_.std()
        dgs.append(eg - ep); nwin += int(eg < ep)
    rep[k_] = {"mean_diff_normMAE_GP_minus_PCHIP": float(np.mean(dgs)), "sd": float(np.std(dgs, ddof=1)), "GP_win_rate": nwin / NREP}
X3["repeated_cv"] = {"n_repeats": NREP, "series": rep}

# C. inversion test on every stress series (knot scan, null simulated from the linear fit, Holm-corrected)
def bstick(T, y):
    best = None
    for kn in T[1:-1]:
        A = np.column_stack([np.ones_like(T), T, np.maximum(T - kn, 0.)]); c3 = np.linalg.lstsq(A, y, rcond=None)[0]
        r3 = y - A @ c3; rss = float(r3 @ r3)
        if best is None or rss < best[0]: best = (rss, float(kn), c3)
    return best

def inv_test(T, y, nsim, rs):
    n = len(T); pl = np.polyfit(T, y, 1); rl = y - np.polyval(pl, T); rss_l = float(rl @ rl)
    rss_b, kn, c3 = bstick(T, y); stat = (rss_l - rss_b) / max(rss_l, 1e-12)
    out = {"n": n, "knot_K": kn, "slope_below": float(c3[1]), "slope_above": float(c3[1] + c3[2]),
           "frac_variance_by_knot": float(stat), "sign_change": bool(c3[1] < 0 < c3[1] + c3[2])}
    if rss_l < 1e-9:
        out["p_knot"] = 1.0; return out
    sgm = np.sqrt(rss_l / (n - 2)); cnt = 0
    for _ in range(nsim):
        ysim = np.polyval(pl, T) + rs.normal(0, sgm, n); rs2 = ysim - np.polyval(np.polyfit(T, ysim, 1), T)
        rl2 = float(rs2 @ rs2); cnt += ((rl2 - bstick(T, ysim)[0]) / max(rl2, 1e-12)) >= stat
    out["p_knot"] = float((1 + cnt) / (1 + nsim)); return out

def holm3(ps):
    ps = np.asarray(ps, float); od = np.argsort(ps); adj = np.zeros(len(ps)); run = 0.
    for r_, i_ in enumerate(od):
        run = max(run, (len(ps) - r_) * ps[i_]); adj[i_] = min(1., run)
    return adj

rs3 = np.random.RandomState(3); inv_rows = []
for al_, pr_, (Tx_, yx_) in XS:
    if len(Tx_) >= 5:
        inv_rows.append({"alloy": al_, "series": pr_, **inv_test(np.asarray(Tx_, float), np.asarray(yx_, float), NSIM, rs3)})
for r_, a_ in zip(inv_rows, holm3([r_["p_knot"] for r_ in inv_rows])): r_["p_holm"] = float(a_)
X3["inversion_test_all_series"] = inv_rows

# D. two-regime slope model: sigma = a + s1 T + (s2 - s1) w softplus((T - T0)/w); T_inv explicit
def tr_model(p, T): return p[0] + p[1] * T + (p[2] - p[1]) * p[4] * np.logaddexp(0., (T - p[3]) / p[4])
def tr_fit(T, y):
    best = None
    for T0s in (100., 150., 220.):
        for w0 in (15., 40.):
            r3 = least_squares(lambda p: tr_model(p, T) - y, [y.mean(), -1.5, 1.5, T0s, w0],
                               bounds=([-3000, -10, -10, 20, 3], [3000, 10, 10, 420, 200]), max_nfev=500)
            if best is None or r3.cost < best.cost: best = r3
    return best.x
def tr_inv(p):
    if not (p[1] < 0 < p[2]): return np.nan
    q = -p[1] / (p[2] - p[1]); return float(p[3] + p[4] * np.log(q / (1 - q)))

pfull = tr_fit(T13, s0); res_f = s0 - tr_model(pfull, T13)
loo3 = np.array([tr_model(tr_fit(T13[np.arange(13) != i], s0[np.arange(13) != i]), T13[i:i + 1])[0] for i in range(13)])
rb3 = np.random.RandomState(5); inv_b, cb3 = [], []
for _ in range(NBOOT3):
    yb = tr_model(pfull, T13) + rb3.choice(res_f * np.sqrt(13 / 8.), 13, replace=True)
    try:
        pb = tr_fit(T13, yb); inv_b.append(tr_inv(pb)); cb3.append(tr_model(pb, Tg))
    except Exception: pass
inv_b = np.array(inv_b); okb = ~np.isnan(inv_b); cb3 = np.array(cb3)
X3["two_regime_model"] = {"params_a_s1_s2_T0_w": pfull.tolist(), "T_inv_K": tr_inv(pfull), "in_sample_RMSE_MPa": float(np.sqrt(np.mean(res_f ** 2))),
    "LOO": metr(s0, loo3), "bootstrap_T_inv_median_K": float(np.median(inv_b[okb])) if okb.any() else None,
    "bootstrap_T_inv_CI95_K": [float(np.percentile(inv_b[okb], 2.5)), float(np.percentile(inv_b[okb], 97.5))] if okb.any() else None,
    "bootstrap_frac_sign_change": float(okb.mean()) if len(inv_b) else None}

# E. PySR seed stability (same LOO one-SE rule, tanh in operator set)
X3["pysr_seeds"] = []
if py is not None:
    for sd_ in ([11] if FAST else [11, 12, 13]):
        try:
            pkx = f"{OUT}/pysr_seed{sd_}.pkl"
            if os.path.exists(pkx):
                mdl = pickle.load(open(pkx, "rb"))
            else:
                kw3 = dict(binary_operators=["+", "-", "*"], unary_operators=["exp", "square", "tanh"], maxsize=12, parsimony=0.001,
                           niterations=PY_IT, populations=20, population_size=40,
                           elementwise_loss="loss(prediction, target) = (prediction - target)^2",
                           verbosity=0, progress=False, temp_equation_file=True)
                try: mdl = PySRRegressor(random_state=sd_, parallelism="serial", deterministic=True, **kw3).fit(X, z)
                except Exception: mdl = PySRRegressor(random_state=sd_, **kw3).fit(X, z)
                pickle.dump(mdl, open(pkx, "wb"))
            eq3 = mdl.equations_.reset_index(drop=True); st3 = []
            for i in range(len(eq3)):
                pr3, p03 = build(mdl.sympy(i)); e3 = np.zeros(13)
                for j in range(13):
                    m3 = np.ones(13, bool); m3[j] = False
                    e3[j] = (pr3(T13[j:j + 1] / 100., fitc(pr3, p03, T13[m3] / 100., z[m3]))[0] - z[j]) * ys_
                st3.append((int(eq3.loc[i, "complexity"]), float(np.mean(e3 ** 2)), float(np.std(e3 ** 2) / np.sqrt(13))))
            ms3 = np.array([s_[1] for s_ in st3]); b3 = int(np.argmin(ms3))
            si = min([i for i in range(len(st3)) if ms3[i] <= ms3[b3] + st3[b3][2]], key=lambda i: st3[i][0])
            pr3, p03 = build(mdl.sympy(si)); pf3 = fitc(pr3, p03, T13 / 100., z); cu3 = pr3(Tg / 100., pf3) * ys_ + ym
            X3["pysr_seeds"].append({"seed": sd_, "equation": str(mdl.sympy(si)), "complexity": st3[si][0],
                "LOO_RMSE_MPa": float(np.sqrt(st3[si][1])),
                "inversion_K": argmin_T(cu3, Tg) if np.all(np.isfinite(cu3)) else None})
        except Exception as e_:
            X3["pysr_seeds"].append({"seed": sd_, "status": f"{type(e_).__name__}: {str(e_)[:100]}"})

# F. Song closure consistency: |dSeff - dS_cal - kappa Qdiss/(2T)| for each model family
Tcl = np.linspace(110, 400, 146)
def _clos(se, ds, qd): return np.abs(se - ds - KAPPA * qd / (2 * Tcl))
cl = {"data_interpolation": float(_clos(np.interp(Tcl, Ts, S), np.interp(Tcl, Tk, Sc), np.interp(Tcl, Tq, Q)).mean()),
      "independent_GPs": float(_clos(pred(GS, Tcl)[0], pred(GSC, Tcl)[0], pred(GQ, Tcl)[0]).mean())}
for nm_, nets_ in [("PINN-data", nets_d), ("PINN-phys", nets_p)]:
    am = members(nets_, Tcl); cl[nm_] = float(np.mean([_clos(am[i][:, 4], am[i][:, 3], am[i][:, 2]).mean() for i in range(len(am))]))
X3["closure_residual_JkgK"] = cl
R["ext3"] = X3

# figure
fig, ax = plt.subplots(2, 3, figsize=(16, 9))
a = ax[0, 0]; nm3 = ["sigma_0", "sigma_Ms", "sigma_Af"]
a.bar(np.arange(3) - .2, [R["stress_loo"][n_]["GP"]["PICP95"] for n_ in nm3], .4, color=PAL["pc"], label="Learned white noise")
a.bar(np.arange(3) + .2, [X3["noise_aware_gp_loo"][n_]["PICP95"] for n_ in nm3], .4, color=PAL["gp"], label="Digitisation-noise GP")
a.axhline(.95, c="k", ls="--", lw=.8); a.set_xticks(range(3)); a.set_xticklabels(nm3); a.set_ylim(0, 1.05)
a.set(ylabel="LOO 95% coverage", title="(a) Interval calibration"); a.legend()
a = ax[0, 1]; lab = [f"{r_['alloy']} {r_['series']}" for r_ in inv_rows]
a.barh(range(len(lab)), [-np.log10(max(r_["p_holm"], 1e-4)) for r_ in inv_rows], color=[PAL["inv"] if r_["sign_change"] else PAL["pc"] for r_ in inv_rows])
a.axvline(-np.log10(.05), c="k", ls="--", lw=.8); a.set_yticks(range(len(lab))); a.set_yticklabels(lab, fontsize=7); a.invert_yaxis()
a.set(xlabel="-log10 Holm-adjusted p (knot test)", title="(b) Slope-change test, all series")
a = ax[0, 2]; zones(a)
if len(cb3): band(a, Tg, np.percentile(cb3, 2.5, 0), np.percentile(cb3, 97.5, 0), PAL["inv"], "Bootstrap 95%")
a.plot(Tg, tr_model(pfull, Tg), c=PAL["inv"], lw=2, label="Two-regime slope model"); a.plot(Tg, mean0, c=PAL["gp"], lw=1.5, ls="--", label="GP mean")
a.scatter(T13, s0, c=PAL["dat"], s=28, zorder=5, label="Data"); a.set(xlabel="T (K)", ylabel="σ$_0$ (MPa)", title=f"(c) Two-regime fit, T$_{{inv}}$={tr_inv(pfull):.0f} K"); a.legend(); a.set_xlim(0, 460)
a = ax[1, 0]; a.bar(list(cl), list(cl.values()), color=[PAL["pc"], PAL["gp"], PAL["pysr"], PAL["pinn"]]); a.tick_params(axis="x", rotation=25)
a.set(ylabel="Mean closure residual (J/kg K)", title="(d) Song closure consistency, 110-400 K")
a = ax[1, 1]; zones(a); sdl = [s_ for s_ in X3["pysr_seeds"] if s_.get("inversion_K") is not None]
if sdl: a.scatter([s_["seed"] for s_ in sdl], [s_["inversion_K"] for s_ in sdl], c=PAL["pysr"], s=70, label="PySR seeds")
a.axhline(R["inversion_GP"]["posterior_median_K"], c=PAL["gp"], lw=2, label="GP median"); a.axhline(tr_inv(pfull), c=PAL["inv"], lw=2, ls="--", label="Two-regime")
a.axhline(150, c="#f4a261", lw=2, ls=":", label="Song 2026"); a.set(xlabel="PySR seed", ylabel="Inversion T (K)", title="(e) Inversion across methods"); a.legend(fontsize=7)
a = ax[1, 2]; ks3 = list(rep); a.bar(range(len(ks3)), [rep[k_]["GP_win_rate"] for k_ in ks3], color=PAL["gp"]); a.axhline(.5, c="k", ls="--", lw=.8)
a.set_xticks(range(len(ks3))); a.set_xticklabels(ks3, rotation=30, ha="right"); a.set(ylabel="Fraction of repeats GP beats PCHIP", title=f"(f) Repeated 4-fold CV ({NREP} repeats)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig10_reviewer_proofing.png", dpi=220); plt.close()

# claims text
XL += ["", "EXTENSION 3: REVIEWER-PROOFING", "-" * 70, "Digitisation-noise GP, LOO (standard GP coverage in brackets):"]
for n_ in nm3:
    v_ = X3["noise_aware_gp_loo"][n_]; XL.append(f"  {n_}: R2={v_['R2']:.4f} MAE={v_['MAE']:.2f} PICP95={v_['PICP95']:.2f} [{R['stress_loo'][n_]['GP']['PICP95']:.2f}]")
XL.append(f"Repeated 4-fold CV ({NREP} repeats), normalised MAE difference GP minus PCHIP and GP win rate:")
for k_, v_ in rep.items(): XL.append(f"  {k_}: {v_['mean_diff_normMAE_GP_minus_PCHIP']:+.4f} (sd {v_['sd']:.4f}) win {v_['GP_win_rate']:.2f}")
XL.append("Slope-change (knot) test per series, Holm-adjusted:")
for r_ in inv_rows:
    XL.append(f"  {r_['alloy']:12s} {r_['series']:20s} n={r_['n']} knot={r_['knot_K']:.0f} K below={r_['slope_below']:+.2f} above={r_['slope_above']:+.2f} "
              f"var.expl={r_['frac_variance_by_knot']:.3f} sign_change={r_['sign_change']} p={r_['p_knot']:.4f} p_holm={r_['p_holm']:.4f}")
XL.append(f"Two-regime model: {X3['two_regime_model']}")
XL.append("PySR seeds: " + "; ".join(str(s_) for s_ in X3["pysr_seeds"]))
XL.append(f"Song closure residual (J/kg K, 110-400 K, kappa={KAPPA:.2f}): {cl}")
XL.append("")

XL.append("")

# OUTPUT
json.dump(R, open(f"{OUT}/results.json", "w"), indent=2, default=float)
iv, cv_, tm, cpg = R["inversion_GP"], R["cc_violation"], R["Tmin_GP"], R["COP_GP"]
L = ["FINAL RESULTS (computed from data/dataset.csv)", "=" * 70,
     f"sigma_0 LOO: GP R2={stress_loo['sigma_0']['GP']['R2']:.4f} MAE={stress_loo['sigma_0']['GP']['MAE']:.2f} MPa | "
     f"PCHIP R2={stress_loo['sigma_0']['PCHIP']['R2']:.4f} MAE={stress_loo['sigma_0']['PCHIP']['MAE']:.2f} | Wilcoxon p={R['wilcoxon_GP_vs_PCHIP_p']}",
     f"Inversion GP: mean curve {iv['mean_curve_K']:.1f} K, posterior median {iv['posterior_median_K']:.1f} K, 95% {iv['posterior_CI95_K'][0]:.0f}-{iv['posterior_CI95_K'][1]:.0f} K",
     f"Data slope negative between {R['data_negative_slope_interval_K']} K; GP mean slope 20-100 K {iv['slope_20_100']:+.2f}, 250-400 K {iv['slope_250_400']:+.2f} MPa/K",
     f"CC violation 50-150 K: mean excess {cv_['window_50_150']['mean']:+.2f} MPa/K, 95% {cv_['window_50_150']['CI95']}, P(neg)={cv_['window_50_150']['P_negative']:.3f}",
     f"CC check 250-400 K: mean excess {cv_['window_250_400']['mean']:+.2f} MPa/K, 95% {cv_['window_250_400']['CI95']}",
     f"Gap test GP: MAE {R['gap_test_GP']['held_MAE']:.1f} MPa, inside 95%: {R['gap_test_GP']['inside_95CI']}, inversion {R['gap_test_GP']['T_inv_K']} K",
     f"Extrapolation GP (train T>=150 K): {R['extrapolation_GP']}",
     f"Gap test PINN: {R['gap_test_PINN']}",
     f"Song relation audit: kappa_fit={KAPPA:.2f}; residual kappa=1 {R['song_audit']['mean_abs_residual_kappa1_JkgK']:.2f}, fitted {R['song_audit']['mean_abs_residual_kappa_fit_JkgK']:.2f} J/kgK",
     f"CC audit 200-450 K: measured {R['cc_audit_classical']['measured_slope_MPaK']:.2f} vs implied {R['cc_audit_classical']['cc_implied_slope_MPaK']:.2f} MPa/K",
     f"Tmin: GP {tm['median']:.1f} K {tm['CI95']}, data {tm['data_linear']:.1f}, PINN {R['Tmin_PINN']['mean']:.1f}+/-{R['Tmin_PINN']['sd_ensemble']:.1f} (ensemble sd), Song 97 K",
     f"COP 294 K: GP {cpg['at_294']:.2f} {cpg['CI95']}, PINN {R['COP_PINN']['at_294_mean']:.2f}, raw {cpg['raw_arithmetic']:.2f}, Song 4.6",
     f"Implied Cp (T>=200 K): {R['Cp_implied_ge200']}", "", "4-fold CV R2 / MAE:"]
for m in MODELS: L.append(f"  {m:10s} " + "  ".join(f"{k}:{cvt[m][k]['R2']:.3f}/{cvt[m][k]['MAE']:.2f}" for k in SER))
L += ["", f"PySR: {R['pysr']}"]
L += [""] + XL
open(f"{OUT}/final_claims.txt", "w").write("\n".join(L)); print("\n".join(L)); print(f"\nSaved to {OUT}/")
