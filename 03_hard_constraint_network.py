"""
03_hard_constraint_network.py : hard-constraint Landau-residual PINN for Ti-Al-Cr.
Stress is the fixed Landau curve plus a small learned residual; Seff, dTad and COP are computed from the learned
entropy, dissipation and kappa(T), never fitted as independent outputs. Deep ensemble, no dropout.
Needs outputs/landau/{maxwell_table.npz, landau_params.json, paper_metrics.json} from 02_landau_fit.py.
Run: python 03_hard_constraint_network.py   (FAST=1 for a smoke test)
"""
import os, json, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
warnings.filterwarnings("ignore")
FAST = os.environ.get("FAST") == "1"; NS = 2 if FAST else 8; EP = 500 if FAST else 3500
DATA = os.environ.get("SMA_DATA", "data/dataset.csv"); OUT = "outputs/pinn_hard"; os.makedirs(OUT, exist_ok=True)
RHO, EPS = 4500., 0.064
df = pd.read_csv(DATA); ta = df[df.alloy_system == "Ti-Al-Cr"]
def rows(p, sp=None):
    s = ta[ta.property == p].dropna(subset=["T_K", "value"]).sort_values("T_K")
    if sp: s = s[s.split.isin(sp)]
    return s.T_K.values.astype(float), s.value.values.astype(float)
tab = np.load("outputs/landau/maxwell_table.npz"); PL = json.load(open("outputs/landau/landau_params.json"))["params"]; DB = json.load(open("outputs/landau/paper_metrics.json"))["debye_cp"]
AM, AC, ALPHA, STILDE = float(tab["AM"]), float(tab["AC"]), tab["ALPHA"], tab["STILDE"]
barrett = lambda T, Ts_: 0.5 * Ts_ / np.tanh(Ts_ / (2 * np.maximum(T, 1e-3)))
nB = lambda T, t: 1. / np.expm1(np.minimum(t / np.maximum(T, 1e-3), 700.))
def sigL(T): z = PL[1] * (barrett(T, PL[3]) - PL[2]) - PL[4] * nB(T, PL[5]); return PL[0] * np.interp(AM + (AC - AM) / (1 + np.exp(-np.clip(z, -50, 50))), ALPHA, STILDE)
dsigL = lambda T: (sigL(T + .5) - sigL(T - .5)) / 1.
_x = np.linspace(1e-6, 60, 60001); _g = _x**4 * np.exp(_x) / np.expm1(_x)**2; _I = np.concatenate([[0], np.cumsum(.5 * (_g[1:] + _g[:-1]) * np.diff(_x))])
A_DP = 3 * 8.314462618 / ((0.7525 * 47.867 + 0.20 * 26.982 + 0.0475 * 51.996) / 1000)
Cp = lambda T: DB["gamma"] * T + A_DP * 3 * (T / DB["theta"])**3 * np.interp(DB["theta"] / np.maximum(T, 1e-6), _x, _I)

tr = lambda p: rows(p, ["train"]); te = lambda p: rows(p, ["test"])
D = {k: tr(k) for k in ("sigma_Ms", "sigma_Af", "sigma_0", "Qdiss", "DeltaSeff")}; D["DeltaS_cal"] = rows("DeltaS_cal")
Tt_, dT_ = rows("dTad_meas"); hold = np.arange(len(Tt_)) % 4 == 3; D["dTad"] = (Tt_[~hold], dT_[~hold]); HELD = {k: te(k) for k in ("sigma_Ms", "sigma_Af", "sigma_0", "Qdiss", "DeltaSeff")}; HELD["dTad"] = (Tt_[hold], dT_[hold])
Tc = np.linspace(260, 400, 40)
def mlp(h=24, o=1): return nn.Sequential(nn.Linear(1, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.r, s.h, s.S, s.Q, s.k = mlp(), mlp(), mlp(), mlp(), mlp(); s.lf = nn.Parameter(torch.tensor(-0.3)); nn.init.zeros_(s.r[-1].weight); nn.init.zeros_(s.r[-1].bias)
    def comps(s, T):
        x = (T / 100.).reshape(-1, 1); sg0 = torch.tensor(sigL(T.numpy()), dtype=torch.float32) + 40. * s.r(x).squeeze(-1); h = 100. * nn.functional.softplus(s.h(x).squeeze(-1))
        dS = -30. * nn.functional.softplus(s.S(x).squeeze(-1)); Q = 1500. * nn.functional.softplus(s.Q(x).squeeze(-1)); kap = .5 + 5.5 * torch.sigmoid(s.k(x).squeeze(-1))
        Se = dS + kap * Q / (2 * T); return sg0, h, dS, Q, Se
T_ = lambda a: torch.tensor(a, dtype=torch.float32); Y_ = lambda k: (T_(D[k][0]), T_(D[k][1]))
def loss_fn(m):
    L = 0.
    for k, f in (("sigma_0", lambda c: c[0]), ("sigma_Ms", lambda c: c[0] + c[1] / 2), ("sigma_Af", lambda c: c[0] - c[1] / 2)):
        T, y = Y_(k); L = L + (((f(m.comps(T)) - y) / 10.) ** 2).mean()
    T, y = Y_("DeltaS_cal"); L = L + (((m.comps(T)[2] - y) / 1.) ** 2).mean()
    T, y = Y_("Qdiss"); L = L + (((m.comps(T)[3] - y) / 60.) ** 2).mean()
    T, y = Y_("DeltaSeff"); L = L + (((m.comps(T)[4] - y) / 1.) ** 2).mean()
    T, y = Y_("dTad"); L = L + (((T * m.comps(T)[4] / T_(Cp(T.numpy())) - y) / .5) ** 2).mean()
    Tg = T_(Tc).requires_grad_(True); x = (Tg / 100.).reshape(-1, 1); r = 40. * m.r(x).squeeze(-1)
    dr = torch.autograd.grad(r.sum(), Tg, create_graph=True)[0]; dS = -30. * nn.functional.softplus(m.S(x).squeeze(-1)); f = torch.exp(m.lf)
    cc = dsigL(Tc).astype(np.float32); L = L + 2. * (((T_(cc) + dr + f * (RHO / EPS) * 1e-6 * dS) / .5) ** 2).mean()
    return L
def train(seed):
    torch.manual_seed(seed); m = Net(); o = torch.optim.Adam(m.parameters(), 4e-3, weight_decay=1e-5); s = torch.optim.lr_scheduler.CosineAnnealingLR(o, EP, 1e-4)
    for _ in range(EP): o.zero_grad(); l = loss_fn(m); l.backward(); nn.utils.clip_grad_norm_(m.parameters(), 5.); o.step(); s.step()
    return m
M = [train(i) for i in range(NS)]
def ens(T):
    T = T_(np.atleast_1d(T));
    with torch.no_grad(): c = [[x.numpy() for x in m.comps(T)] for m in M]
    return [np.mean([a[i] for a in c], 0) for i in range(5)], [np.std([a[i] for a in c], 0) for i in range(5)]
mape = lambda y, p: float(np.mean(np.abs(p - y) / np.abs(y)) * 100); R = {}
def ev(k, f):
    T, y = HELD[k]
    if len(T) == 0: return
    (s0, h, dS, Q, Se), _ = ens(T); p = f(T, s0, h, dS, Q, Se); R[k] = {"MAPE": mape(y, p), "MAE": float(np.mean(np.abs(p - y))), "n_points": int(len(T))}
ev("sigma_Ms", lambda T, s0, h, dS, Q, Se: s0 + h / 2); ev("sigma_Af", lambda T, s0, h, dS, Q, Se: s0 - h / 2); ev("sigma_0", lambda T, s0, h, dS, Q, Se: s0)
ev("Qdiss", lambda T, s0, h, dS, Q, Se: Q); ev("DeltaSeff", lambda T, s0, h, dS, Q, Se: Se); ev("dTad", lambda T, s0, h, dS, Q, Se: T * Se / Cp(T))
Tz = np.linspace(50, 160, 1101); (s0, h, dS, Q, Se), _ = ens(Tz); i = np.where((Se[:-1] > 0) & (Se[1:] <= 0))[0]
tmin = float(Tz[i[0]] + (Tz[i[0] + 1] - Tz[i[0]]) * Se[i[0]] / (Se[i[0]] - Se[i[0] + 1])) if i.size else float("nan")
zs = []
for m in M:
    with torch.no_grad(): se = m.comps(T_(Tz))[4].numpy()
    j = np.where((se[:-1] > 0) & (se[1:] <= 0))[0]; zs.append(float(Tz[j[0]]) if j.size else np.nan)
(_, _, dS2, Q2, Se2), _ = ens([294., 297.]); cop = lambda T, S, Q_: float(T * abs(S) / Q_)
R["T_min_K"] = {"ensemble_mean_curve": tmin, "member_mean": float(np.nanmean(zs)), "member_sd": float(np.nanstd(zs)), "experiment": 97., "deviation_pct": abs(tmin - 97.) / 97. * 100}
R["COP_294K"] = {"model": cop(294., Se2[0], Q2[0]), "experiment": 4.6, "deviation_pct": abs(cop(294., Se2[0], Q2[0]) - 4.6) / 4.6 * 100}
R["COP_297K"] = {"model": cop(297., Se2[1], Q2[1]), "experiment": 4.6, "deviation_pct": abs(cop(297., Se2[1], Q2[1]) - 4.6) / 4.6 * 100}
R["conversion_f_mean"] = float(np.mean([float(torch.exp(m.lf)) for m in M]))
Tgrid = np.arange(20., 461., 5.); (s0, h, dS, Q, Se), sd = ens(Tgrid)
pd.DataFrame({"T_K": Tgrid, "sigma_0": s0, "sigma_0_sd": sd[0], "sigma_Ms": s0 + h / 2, "sigma_Af": s0 - h / 2, "dS": dS, "Qdiss": Q, "DeltaSeff": Se, "dTad": Tgrid * Se / Cp(Tgrid)}).to_csv(f"{OUT}/pinn_hard_predictions.csv", index=False)
json.dump(R, open(f"{OUT}/pinn_hard_metrics.json", "w"), indent=1)
print("HARD-CONSTRAINT PINN, held-out results (ensemble of", NS, ")"); [print(f"  {k:10s} MAPE {v['MAPE']:6.2f}%  MAE {v['MAE']:.3f}  (points {v['n_points']})") for k, v in R.items() if isinstance(v, dict) and "MAPE" in v]
print(f"  T_min {tmin:.1f} K (member mean {R['T_min_K']['member_mean']:.1f} +/- {R['T_min_K']['member_sd']:.1f}; experiment 97 K, deviation {R['T_min_K']['deviation_pct']:.1f}%)")
print(f"  COP 294 K {R['COP_294K']['model']:.2f} (dev {R['COP_294K']['deviation_pct']:.1f}%), 297 K {R['COP_297K']['model']:.2f} (dev {R['COP_297K']['deviation_pct']:.1f}%) | f={R['conversion_f_mean']:.3f}")
