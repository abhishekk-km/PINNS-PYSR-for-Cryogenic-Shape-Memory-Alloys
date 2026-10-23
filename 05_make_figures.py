"""
05_make_figures.py : manuscript figures for the Ti-Al-Cr paper. Run from the repository root: python 05_make_figures.py
Reads data/dataset.csv and the outputs of 02_landau_fit.py, 03_hard_constraint_network.py and 01_symbolic_regression.py.
Writes figures/*.pdf and figures/*.png.
"""
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg"); import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator as PCHIP
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel, DotProduct
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
DATA = os.environ.get("SMA_DATA", "data/dataset.csv"); OUT = "figures"; os.makedirs(OUT, exist_ok=True)
RHO, EPS = 4500., 0.064
K = dict(red="#E64B35", cyan="#4DBBD5", green="#00A087", blue="#3C5488", orange="#F39B7F", slate="#8491B4", mint="#91D1C2", crim="#DC0000", brown="#7E6148", grey="#B09C85")
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.direction": "out", "ytick.direction": "out", "lines.linewidth": 1.2, "lines.markersize": 3.2, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "legend.frameon": False, "axes.grid": False, "figure.constrained_layout.use": True})
MM = 1 / 25.4; W2, W1 = 183 * MM, 89 * MM
def tag(ax, s, dx=-0.17): ax.text(dx, 1.05, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
def save(fig, name): fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png"); plt.close(fig); print("saved", f"{OUT}/{name}.pdf")
def jload(p, d=None):
    try: return json.load(open(p))
    except Exception: return d if d is not None else {}

df = pd.read_csv(DATA); ta = df[df.alloy_system == "Ti-Al-Cr"]
def ser(p, sp=None):
    s = ta[ta.property == p].dropna(subset=["T_K", "value"]).sort_values("T_K")
    if sp: s = s[s.split.isin(sp)]
    return s.T_K.values.astype(float), s.value.values.astype(float), s.split.values
T0_, S0, SP0 = ser("sigma_0", ["train", "test"]); TMs, SMs, SPMs = ser("sigma_Ms", ["train", "test"])
Tq, Q, _ = ser("Qdiss"); Ts, SE, _ = ser("DeltaSeff"); Td, DS, _ = ser("DeltaS_cal"); Tt, DT, _ = ser("dTad_meas")
PM = jload("outputs/landau/paper_metrics.json"); LP = jload("outputs/landau/landau_params.json"); PL = np.array(LP["params"])
tab = np.load("outputs/landau/maxwell_table.npz"); AM, AC, ALPHA, STILDE = float(tab["AM"]), float(tab["AC"]), tab["ALPHA"], tab["STILDE"]
barrett = lambda T, t: 0.5 * t / np.tanh(t / (2 * np.maximum(T, 1e-3)))
nB = lambda T, t: 1. / np.expm1(np.minimum(t / np.maximum(T, 1e-3), 700.))
zch = lambda T: PL[1] * (barrett(T, PL[3]) - PL[2]); zel = lambda T: -PL[4] * nB(T, PL[5])
alz = lambda z: AM + (AC - AM) / (1 + np.exp(-np.clip(z, -50, 50)))
sigL = lambda T: PL[0] * np.interp(alz(zch(T) + zel(T)), ALPHA, STILDE)
Tg = np.linspace(20, 460, 441)
def gpfit(T, y, lin=True):
    sx, sy = StandardScaler(), StandardScaler(); X = sx.fit_transform(T.reshape(-1, 1)); z = sy.fit_transform(y.reshape(-1, 1)).ravel()
    k = C(1., (1e-2, 1e2)) * Matern(1., (.2, 10.), nu=2.5) + WhiteKernel(1e-3, (1e-6, 1e-1))
    if lin: k = C(1., (1e-2, 1e2)) * DotProduct(sigma_0=1., sigma_0_bounds=(1e-2, 1e2)) + k
    g = GPR(k, n_restarts_optimizer=8, alpha=1e-8, random_state=42).fit(X, z)
    def f(t):
        m, s = g.predict(sx.transform(np.atleast_1d(t).reshape(-1, 1)), return_std=True); return sy.inverse_transform(m.reshape(-1, 1)).ravel(), s * sy.scale_[0]
    return f
infl = PM.get("conformal", {}).get("GP", {}).get("inflation", 1.0)
gp0 = gpfit(T0_, S0); mu, sd = gp0(Tg); band = 1.96 * sd * infl
BR = (110, 180)

# fig1_stress_landau
fig, ax = plt.subplots(2, 2, figsize=(W2, 0.66 * W2)); a = ax[0, 0]
a.axvspan(*BR, color=K["grey"], alpha=.15, lw=0); a.fill_between(Tg, mu - band, mu + band, color=K["blue"], alpha=.18, lw=0)
a.plot(Tg, mu, color=K["blue"], label="Gaussian process"); a.plot(Tg, sigL(Tg), color=K["red"], label="Landau (Maxwell)")
tr = SP0 == "train"; a.plot(T0_[tr], S0[tr], "o", color="k", mfc="k", label="train"); a.plot(T0_[~tr], S0[~tr], "o", color="k", mfc="white", label="held out")
a.set(xlabel="Temperature (K)", ylabel=r"$\sigma_0$ (MPa)", xlim=(0, 470)); a.legend(loc="upper left", ncol=2, columnspacing=.8, handlelength=1.4); tag(a, "a")
a = ax[0, 1]; a.axvspan(*BR, color=K["grey"], alpha=.15, lw=0); a.axhline(0, color="k", lw=.5)
sl_g = np.gradient(mu, Tg); sl_l = np.gradient(sigL(Tg), Tg); a.plot(Tg, sl_g, color=K["blue"]); a.plot(Tg, sl_l, color=K["red"])
Tm_ = .5 * (T0_[1:] + T0_[:-1]); a.plot(Tm_, np.diff(S0) / np.diff(T0_), "s", color=K["green"], ms=2.6, label="finite difference")
for s_, c_ in ((sl_g, K["blue"]), (sl_l, K["red"])):
    i = np.where((s_[:-1] < 0) & (s_[1:] >= 0))[0]
    if i.size: a.plot(Tg[i[0]], 0, "v", color=c_, ms=4)
a.set(xlabel="Temperature (K)", ylabel=r"d$\sigma_0$/d$T$ (MPa K$^{-1}$)", xlim=(0, 470)); a.legend(loc="lower right"); tag(a, "b")
a = ax[1, 0]; zc, ze = zch(Tg), zel(Tg); a.axhline(0, color="k", lw=.5)
a.plot(Tg, zc, color=K["red"], label="order-parameter channel"); a.plot(Tg, ze, color=K["blue"], label="elastic channel"); a.plot(Tg, zc + ze, color="k", label=r"$z(T)$")
dz = np.gradient(zc + ze, Tg); i = np.where((dz[:-1] < 0) & (dz[1:] >= 0))[0]
if i.size: a.axvline(Tg[i[0]], color=K["grey"], ls="--", lw=.8)
a.set(xlabel="Temperature (K)", ylabel="Landau driving function", xlim=(0, 470)); a.legend(loc="upper left"); tag(a, "c")
a = ax[1, 1]; a.plot(ALPHA, STILDE, color=K["green"]); aT = alz(zc + ze); sT = np.interp(aT, ALPHA, STILDE)
a.plot(aT, sT, color=K["crim"], lw=2.2, solid_capstyle="round", label="range explored by the fit")
a.axvline(AM, color=K["grey"], ls=":", lw=.8); a.axvline(AC, color=K["grey"], ls=":", lw=.8)
a.text(AM, a.get_ylim()[1] * .02, r" $\alpha_M$", fontsize=6.5, va="bottom"); a.text(AC, a.get_ylim()[1] * .02, r"$\alpha_C$ ", fontsize=6.5, ha="right", va="bottom")
a.set(xlabel=r"Reduced coefficient $\alpha=ac/b^{2}$", ylabel=r"Reduced stress $s(\alpha)$"); a.legend(loc="upper left"); tag(a, "d"); save(fig, "fig1_stress_landau")

# fig2_symbolic_regression
pr = pd.DataFrame(); x_c = 100.
try:
    import sympy as sp
    pr = pd.read_csv("outputs/pysr_equations_pareto.csv"); row = pr.loc[pr["score"].idxmax()]; x0 = sp.Symbol("x0")
    f_sym = sp.lambdify(x0, sp.sympify(str(row["equation"]), locals={"square": lambda z: z**2}), "numpy"); trn = SPMs == "train"
    mS, sS = SMs[trn].mean(), SMs[trn].std(); cub = lambda T: f_sym(np.asarray(T, float) / x_c) * sS + mS
except Exception:
    cub = lambda T: 659.33 - 5.4357 * np.asarray(T) + 2.2062e-2 * np.asarray(T)**2 - 2.2385e-5 * np.asarray(T)**3
fig, ax = plt.subplots(1, 3, figsize=(W2, 0.34 * W2)); a = ax[0]
if len(pr):
    p2 = pr.sort_values("complexity"); a.plot(p2.complexity, p2.RMSE_MPa, "-o", color=K["slate"]); sel = pr.loc[pr["score"].idxmax()]
    a.plot(sel.complexity, sel.RMSE_MPa, "o", color=K["crim"], ms=6, mfc="none", mew=1.2); a.set_yscale("log")
a.set(xlabel="Expression complexity", ylabel="RMSE (MPa)"); tag(a, "a")
a = ax[1]; trn = SPMs == "train"; Tc_ = np.linspace(20, 460, 441); cc = cub(Tc_)
a.plot(Tc_, cc, color=K["red"]); a.plot(TMs[trn], SMs[trn], "o", color="k"); a.plot(TMs[~trn], SMs[~trn], "o", color="k", mfc="white")
i = int(np.argmin(cc[(Tc_ > 60) & (Tc_ < 300)])); Tmin_ = Tc_[(Tc_ > 60) & (Tc_ < 300)][i]; a.axvline(Tmin_, color=K["grey"], ls="--", lw=.8)
a.set(xlabel="Temperature (K)", ylabel=r"$\sigma_{\mathrm{Ms}}$ (MPa)"); tag(a, "b")
a = ax[2]; res = SMs - cub(TMs); a.axhline(0, color="k", lw=.5); a.bar(TMs[trn], res[trn], width=9, color=K["slate"]); a.bar(TMs[~trn], res[~trn], width=9, color=K["orange"])
a.set(xlabel="Temperature (K)", ylabel="Residual (MPa)"); tag(a, "c"); save(fig, "fig2_symbolic_regression")

# fig3_entropy_dissipation
Ti = np.arange(max(Ts.min(), Td.min(), Tq.min()), min(Ts.max(), Td.max(), Tq.max()) + 1, 2.); dSi, Qi, Sei = PCHIP(Td, DS)(Ti), PCHIP(Tq, Q)(Ti), PCHIP(Ts, SE)(Ti)
kap = PM.get("kappa", {}).get("fit", 2.8); dSL = -(EPS / RHO) * 1e6 * np.gradient(sigL(Tg), Tg)
fig, ax = plt.subplots(1, 3, figsize=(W2, 0.34 * W2)); a = ax[0]; a.axhline(0, color="k", lw=.5)
a.plot(Td, DS, "o", color="k", label="calorimetry"); a.plot(Tg, dSL, color=K["red"], label="Landau (CC)")
m = (Tg >= Td.min()) & (Tg <= Td.max()); a.fill_between(Tg[m], np.interp(Tg[m], Td, DS), dSL[m], color=K["cyan"], alpha=.35, lw=0, label="excess")
a.set(xlabel="Temperature (K)", ylabel=r"$\Delta S$ (J kg$^{-1}$ K$^{-1}$)"); a.legend(loc="lower right"); tag(a, "a")
a = ax[1]; a.axhline(0, color="k", lw=.5); a.plot(Ts, SE, "o", color="k", label="measured")
a.plot(Ti, dSi + Qi / (2 * Ti), color=K["slate"], label=r"$\kappa=1$"); a.plot(Ti, dSi + kap * Qi / (2 * Ti), color=K["red"], label=rf"$\kappa={kap:.1f}$")
a.set(xlabel="Temperature (K)", ylabel=r"$\Delta S_{\mathrm{eff}}$ (J kg$^{-1}$ K$^{-1}$)"); a.legend(loc="lower right"); tag(a, "b")
a = ax[2]; ok = np.abs(DT) >= 1.; Cpi = Tt[ok] * np.interp(Tt[ok], Ts, SE) / DT[ok]; dbm = PM.get("debye_cp", {})
_x = np.linspace(1e-6, 60, 60001); _g = _x**4 * np.exp(_x) / np.expm1(_x)**2; _I = np.concatenate([[0], np.cumsum(.5 * (_g[1:] + _g[:-1]) * np.diff(_x))])
A_DP = 3 * 8.314462618 / ((0.7525 * 47.867 + 0.20 * 26.982 + 0.0475 * 51.996) / 1000)
Cpf = lambda T: dbm.get("gamma", .16) * T + A_DP * 3 * (T / dbm.get("theta", 274.))**3 * np.interp(dbm.get("theta", 274.) / np.maximum(T, 1e-6), _x, _I)
a.plot(Tt[ok], Cpi, "o", color="k", label="implied"); a.plot(Tg, Cpf(Tg), color=K["red"], label="Debye + electronic"); a.axhline(588, color=K["slate"], ls="--", lw=.9, label="constant")
a.set(xlabel="Temperature (K)", ylabel=r"$C_p$ (J kg$^{-1}$ K$^{-1}$)"); a.legend(loc="lower right"); tag(a, "c"); save(fig, "fig3_entropy_dissipation")

# fig4_caloric_chain
PH = None
try: PH = pd.read_csv("outputs/pinn_hard/pinn_hard_predictions.csv")
except Exception: pass
gS, gQ = gpfit(Ts, SE, lin=False), gpfit(Tq, Q, lin=False); Tx = np.linspace(max(Ts.min(), Tq.min()), min(Ts.max(), Tq.max()), 300)
fig, ax = plt.subplots(1, 3, figsize=(W2, 0.34 * W2)); a = ax[0]; a.axhline(0, color="k", lw=.5); mS_, sS_ = gS(Tx)
a.fill_between(Tx, mS_ - 1.96 * sS_, mS_ + 1.96 * sS_, color=K["blue"], alpha=.2, lw=0, label="Gaussian process"); a.plot(Ts, SE, "o", color="k", label="measured")
if PH is not None: a.plot(PH.T_K, PH.DeltaSeff, color=K["red"], label="hard-constraint network")
a.set(xlabel="Temperature (K)", ylabel=r"$\Delta S_{\mathrm{eff}}$ (J kg$^{-1}$ K$^{-1}$)", xlim=(30, 420)); a.legend(loc="upper right"); tag(a, "a")
a = ax[1]; a.axhline(0, color="k", lw=.5); a.plot(Tt, DT, "o", color="k", label="measured")
if PH is not None: a.plot(PH.T_K, PH.dTad, color=K["red"], label=r"$T\Delta S_{\mathrm{eff}}/C_p(T)$")
a.plot(Tx, Tx * gS(Tx)[0] / 588., color=K["slate"], ls="--", lw=.9, label=r"constant $C_p$")
a.set(xlabel="Temperature (K)", ylabel=r"$\Delta T_{\mathrm{ad}}$ (K)", xlim=(30, 420)); a.legend(loc="upper right"); tag(a, "b")
a = ax[2]; Tc2 = np.linspace(110, 402, 300); sg, ssg = gS(Tc2); qg, _ = gQ(Tc2); cop = Tc2 * np.abs(sg) / qg; lo = Tc2 * np.abs(sg + 1.96 * ssg) / qg; hi = Tc2 * np.abs(sg - 1.96 * ssg) / qg
a.fill_between(Tc2, np.minimum(lo, hi), np.maximum(lo, hi), color=K["blue"], alpha=.2, lw=0, label="Gaussian process"); a.plot(Tc2, cop, color=K["blue"])
if PH is not None:
    p = PH[(PH.DeltaSeff < 0) & (PH.T_K <= 402) & (PH.T_K >= 110)]; a.plot(p.T_K, p.T_K * p.DeltaSeff.abs() / p.Qdiss, color=K["red"], label="hard-constraint network")
a.plot(297, 4.6, "*", color=K["crim"], ms=7, label="experiment"); a.set(xlabel="Temperature (K)", ylabel=r"COP$_{\mathrm{mat}}$", ylim=(0, None)); a.legend(loc="lower right"); tag(a, "c"); save(fig, "fig4_caloric_chain")

# fig5_model_parity
fig, ax = plt.subplots(1, 2, figsize=(W2, 0.42 * W2)); a = ax[0]; lo_, hi_ = S0.min() * .9, S0.max() * 1.1; xx = np.array([lo_, hi_])
a.fill_between(xx, xx * .95, xx * 1.05, color=K["grey"], alpha=.2, lw=0); a.plot(xx, xx, color="k", lw=.5)
for nm, c_, mk in (("GP", K["blue"], "o"), ("Landau-GP", K["green"], "s"), ("Landau", K["red"], "^")):
    pr_ = PM.get("loo", {}).get(nm, {}).get("pred")
    if pr_ is not None and len(pr_) == len(S0): a.plot(S0, pr_, mk, color=c_, mfc="none", label="Gaussian process" if nm == "GP" else nm)
a.set(xlabel=r"Measured $\sigma_0$ (MPa)", ylabel="Leave-one-out prediction (MPa)", xlim=(lo_, hi_), ylim=(lo_, hi_)); a.legend(loc="upper left"); a.set_aspect("equal", "box"); tag(a, "a")
a = ax[1]; a.axhspan(95, 105, color=K["grey"], alpha=.2, lw=0); a.axhline(100, color="k", lw=.5)
if PH is not None:
    sp_ = lambda p: ta[(ta.property == p) & (ta.split == "test")].sort_values("T_K")
    items = [("sigma_Ms", r"$\sigma_{\mathrm{Ms}}$", K["blue"], "sigma_Ms"), ("sigma_Af", r"$\sigma_{\mathrm{Af}}$", K["cyan"], "sigma_Af"), ("sigma_0", r"$\sigma_0$", K["green"], "sigma_0"),
             ("Qdiss", r"$Q_{\mathrm{diss}}$", K["orange"], "Qdiss"), ("DeltaSeff", r"$\Delta S_{\mathrm{eff}}$", K["red"], "DeltaSeff")]
    for col, lab, c_, prop in items:
        r_ = sp_(prop)
        if len(r_): a.plot(r_.T_K, np.interp(r_.T_K, PH.T_K, PH[col]) / r_.value.values * 100, "o", color=c_, label=lab)
    hold = np.arange(len(Tt)) % 4 == 3
    if hold.any(): a.plot(Tt[hold], np.interp(Tt[hold], PH.T_K, PH.dTad) / DT[hold] * 100, "s", color=K["brown"], label=r"$\Delta T_{\mathrm{ad}}$")
    a.legend(loc="best", ncol=3, columnspacing=.8, handletextpad=.3)
a.set(xlabel="Temperature (K)", ylabel="Prediction / measurement (%)"); tag(a, "b"); save(fig, "fig5_model_parity")
