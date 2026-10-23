"""
06_overview_figure.py : overview, Landau-landscape and kappa-map figures. Run after 02_landau_fit.py:
python 06_overview_figure.py. Reads data/dataset.csv and outputs/landau/*; writes figures/fig0, fig6, fig7.
"""
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from scipy.interpolate import PchipInterpolator as PCHIP
warnings.filterwarnings("ignore")

DATA = os.environ.get("SMA_DATA", "data/dataset.csv"); OUT = "figures"; os.makedirs(OUT, exist_ok=True)
RHO, EPS = 4500., 0.064
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7,
    "axes.labelsize": 7.5, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "savefig.dpi": 600,
    "legend.frameon": False, "figure.constrained_layout.use": True})
MM = 1 / 25.4; W2 = 183 * MM
def tag(ax, s, dx=-0.14): ax.text(dx, 1.05, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
def save(fig, name): fig.savefig(f"{OUT}/{name}.pdf"); fig.savefig(f"{OUT}/{name}.png"); plt.close(fig); print("saved", f"{OUT}/{name}.pdf")

# data + Landau model (same definitions as 02_landau_fit.py)
df = pd.read_csv(DATA); ta = df[df.alloy_system == "Ti-Al-Cr"]
def ser(p, sp=None):
    s = ta[ta.property == p].dropna(subset=["T_K", "value"]).sort_values("T_K")
    if sp: s = s[s.split.isin(sp)]
    return s.T_K.values.astype(float), s.value.values.astype(float)
T0_, S0 = ser("sigma_0", ["train", "test"]); Tq, Q = ser("Qdiss"); Ts, SE = ser("DeltaSeff"); Td, DS = ser("DeltaS_cal"); Tt, DT = ser("dTad_meas")
PL = np.array(json.load(open("outputs/landau/landau_params.json"))["params"])
tab = np.load("outputs/landau/maxwell_table.npz"); AM, AC, ALPHA, STILDE = float(tab["AM"]), float(tab["AC"]), tab["ALPHA"], tab["STILDE"]
try: PM = json.load(open("outputs/landau/paper_metrics.json"))
except Exception: PM = {}
barrett = lambda T, t: 0.5 * t / np.tanh(t / (2 * np.maximum(T, 1e-3)))
nB = lambda T, t: 1. / np.expm1(np.minimum(t / np.maximum(T, 1e-3), 700.))
zT = lambda T: PL[1] * (barrett(T, PL[3]) - PL[2]) - PL[4] * nB(T, PL[5])
alz = lambda z: AM + (AC - AM) / (1 + np.exp(-np.clip(z, -50, 50)))
alpha_T = lambda T: alz(zT(T))
sigL = lambda T: PL[0] * np.interp(alpha_T(T), ALPHA, STILDE)
Tg = np.linspace(20, 410, 600)
KAP = PM.get("kappa", {}).get("fit", 2.8)
dbm = PM.get("debye_cp", {})

# colour maps
COLD_HOT = "coolwarm"
CRYO = LinearSegmentedColormap.from_list("cryo", ["#0b1d51", "#2a52be", "#4DBBD5", "#91D1C2", "#f9e784", "#F39B7F", "#E64B35"])
WELL = "turbo"

def Tmin_cross(Tarr, Sarr):
    i = np.where((Sarr[:-1] > 0) & (Sarr[1:] <= 0))[0]
    return float(Tarr[i[0]] + (Tarr[i[0] + 1] - Tarr[i[0]]) * Sarr[i[0]] / (Sarr[i[0]] - Sarr[i[0] + 1])) if i.size else np.nan

# panel builders
def p_stress(a):
    """sigma_0(T) as one curve coloured by its own slope: blue = reversed, red = classical."""
    y = sigL(Tg); sl = np.gradient(y, Tg)
    pts = np.array([Tg, y]).T.reshape(-1, 1, 2); seg = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap=COLD_HOT, norm=TwoSlopeNorm(0, -max(abs(sl.min()), 1e-3), max(sl.max(), 1e-3)), linewidth=3.2, zorder=3)
    lc.set_array(.5 * (sl[:-1] + sl[1:])); a.add_collection(lc)
    a.fill_between(Tg, y, y.min() - 25, where=sl < 0, color="#4DBBD5", alpha=.20, lw=0)
    a.fill_between(Tg, y, y.min() - 25, where=sl >= 0, color="#E64B35", alpha=.14, lw=0)
    a.scatter(T0_, S0, s=22, c="white", edgecolor="k", linewidth=.7, zorder=5, label="digitised data")
    i = np.where((sl[:-1] < 0) & (sl[1:] >= 0))[0]
    if i.size:
        ti = Tg[i[0]]; yi = float(np.interp(ti, Tg, y))
        a.axvline(ti, color="k", ls=":", lw=.8, zorder=2)
        a.plot(ti, yi, "o", ms=4.5, mfc="gold", mec="k", mew=.7, zorder=6)
        a.annotate(rf"$T_{{inv}}\approx{ti:.0f}$ K", xy=(ti, yi), xytext=(ti + 8, 335),
                   ha="left", va="center", fontsize=7, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=.9),
                   arrowprops=dict(arrowstyle="-", color="k", lw=.6, shrinkA=2, shrinkB=3), zorder=7)
    a.text(55, y.max() * .93, r"$d\sigma_0/dT<0$", color="#2a52be", fontsize=7.5, fontweight="bold")
    a.text(255, y.min() + 10, r"$d\sigma_0/dT>0$", color="#c0392b", fontsize=7.5, fontweight="bold")
    a.set(xlim=(0, 420), ylim=(y.min() - 25, S0.max() * 1.08), xlabel="Temperature (K)", ylabel=r"$\sigma_0$ (MPa)")
    a.legend(loc="upper right")

def landscape_data(Tl, y):
    F = np.empty((len(y), len(Tl)))
    for j, T in enumerate(Tl):
        al = alpha_T(T); s = np.interp(al, ALPHA, STILDE)
        f = al * y**2 / 2 - y**4 / 4 + y**6 / 6 - s * y; F[:, j] = f - f.min()
    return F

def p_landscape(a, cax=None, fig=None):
    """Landau free energy at the Maxwell stress: the double well that exists at every T."""
    Tl = np.linspace(20, 410, 300); y = np.linspace(-.15, 1.35, 400); F = landscape_data(Tl, y)
    m = a.pcolormesh(Tl, y, np.clip(F, 0, .06), cmap=WELL, shading="gouraud", rasterized=True)
    a.contour(Tl, y, F, levels=[.005, .015, .03], colors="white", linewidths=.35, alpha=.7)
    a.set(xlabel="Temperature (K)", ylabel=r"Order parameter $y$ (reduced)", xlim=(20, 410))
    if fig is not None: cb = fig.colorbar(m, ax=a, pad=.01, shrink=.9); cb.set_label(r"$f-f_{\min}$ (reduced)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    return m

def p_kappa(a, fig=None):
    """Effective entropy over (T, kappa): cooling (blue) vs heating (red); white line = cooling onset."""
    Ti = np.arange(max(Ts.min(), Td.min(), Tq.min()), min(Ts.max(), Td.max(), Tq.max()) + 1, 2.)
    dS, Qi = PCHIP(Td, DS)(Ti), PCHIP(Tq, Q)(Ti); kap = np.linspace(.5, 6, 220)
    Se = dS[None, :] + kap[:, None] * Qi[None, :] / (2 * Ti[None, :])
    m = a.pcolormesh(Ti, kap, Se, cmap=COLD_HOT, norm=TwoSlopeNorm(0, Se.min(), Se.max()), shading="gouraud", rasterized=True)
    a.contour(Ti, kap, Se, levels=[0], colors="white", linewidths=1.6)
    a.axhline(1, color="k", ls="--", lw=.7); a.text(Ti.max(), 1.06, r"$\kappa=1$ (Song)", ha="right", fontsize=6)
    a.axhline(KAP, color="k", ls="-", lw=.9); a.text(Ti.max(), KAP + .08, rf"$\kappa={KAP:.1f}$ (fitted)", ha="right", fontsize=6.5, fontweight="bold")
    k0 = Tmin_cross(Ti, dS + KAP * Qi / (2 * Ti))
    if np.isfinite(k0): a.plot(k0, KAP, "*", color="gold", mec="k", ms=10, zorder=6); a.text(k0 + 8, KAP - .45, rf"$T_{{min}}={k0:.0f}$ K", fontsize=6.5, fontweight="bold")
    a.set(xlabel="Temperature (K)", ylabel=r"Dissipation amplification $\kappa$")
    if fig is not None: cb = fig.colorbar(m, ax=a, pad=.01, shrink=.9); cb.set_label(r"$\Delta S_{\rm eff}$ (J kg$^{-1}$K$^{-1}$)", fontsize=6.5); cb.ax.tick_params(labelsize=6)
    return m

def p_cool(a):
    """Measured adiabatic temperature change: lollipops coloured by temperature + Debye-Cp prediction."""
    ok = np.argsort(Tt); T_, D_ = Tt[ok], DT[ok]
    a.axhline(0, color="k", lw=.5)
    for t, d in zip(T_, D_): a.vlines(t, 0, d, color=CRYO((t - 40) / 380), lw=2.2, alpha=.9)
    a.scatter(T_, D_, c=T_, cmap=CRYO, vmin=40, vmax=420, s=34, edgecolor="k", linewidth=.5, zorder=4, label="measured")
    if dbm:
        _x = np.linspace(1e-6, 60, 60001); _g = _x**4 * np.exp(_x) / np.expm1(_x)**2; _I = np.concatenate([[0], np.cumsum(.5 * (_g[1:] + _g[:-1]) * np.diff(_x))])
        A = 3 * 8.314462618 / ((0.7525 * 47.867 + 0.20 * 26.982 + 0.0475 * 51.996) / 1000); th = dbm["theta"]
        Cp = lambda T: dbm["gamma"] * T + A * 3 * (T / th)**3 * np.interp(th / np.maximum(T, 1e-6), _x, _I)
        Tf = np.linspace(Ts.min(), Ts.max(), 300); a.plot(Tf, Tf * np.interp(Tf, Ts, SE) / Cp(Tf), color="k", lw=1.1, label=r"$T\Delta S_{\rm eff}/C_p(T)$, Debye")
        a.plot(Tf, Tf * np.interp(Tf, Ts, SE) / 588., color="grey", lw=.9, ls="--", label=r"constant $C_p$")
    a.set(xlabel="Temperature (K)", ylabel=r"$\Delta T_{\rm ad}$ (K)"); a.legend(loc="lower left")

# hero figure
fig, ax = plt.subplots(2, 2, figsize=(W2, 0.64 * W2))
p_stress(ax[0, 0]); tag(ax[0, 0], "a")
p_landscape(ax[0, 1], fig=fig); tag(ax[0, 1], "b")
p_kappa(ax[1, 0], fig=fig); tag(ax[1, 0], "c")
p_cool(ax[1, 1]); tag(ax[1, 1], "d")
save(fig, "fig0_overview")

# standalone versions
fig, a = plt.subplots(figsize=(W2 * .62, .40 * W2)); p_landscape(a, fig=fig); save(fig, "fig6_landscape")
fig, a = plt.subplots(figsize=(W2 * .62, .40 * W2)); p_kappa(a, fig=fig); save(fig, "fig7_kappa_map")
