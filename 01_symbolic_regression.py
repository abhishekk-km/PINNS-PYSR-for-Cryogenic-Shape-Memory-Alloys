"""
01_symbolic_regression.py : symbolic regression (PySR) of the Ti-Al-Cr upper-plateau stress sigma_Ms(T), with linear baseline,
leave-one-out equation selection, bootstrap bands, caloric chain from measured data and a calorimetric
Clausius-Clapeyron integration test. Reads data/dataset.csv, writes outputs/.
Run: python 01_symbolic_regression.py   (Julia is installed by PySR on first use; delete outputs/pysr_tiAlCr_model.pkl to refit)
"""
# 0. IMPORTS
import os
import time
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import curve_fit, least_squares

warnings.filterwarnings("ignore")
np.random.seed(42)

# 1. PATHS AND CONFIGURATION
DATA_PATH = "data/dataset.csv"
OUT_DIR   = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Physical constants (all from Song 2025/2026 and dataset)
EPS_TR      = 0.064
RHO         = 4500.0
CP          = 588.0     # J/kg·K, used only for the 294 K consistency check
CC_INV_T    = 150.0     # K   sign-change temperature        Song 2026 Fig 1b (compression)
T_MIN_DATA  = 25.0
T_MAX_PHYSICAL = 402.0  # K - experimental limit for reversible operation

# niterations: number of generations the evolutionary algorithm runs
# Higher = better equations, longer runtime
PYSR_NITERATIONS =500   # Set to 500 for final paper run; 100 for testing

# Palette: colour-blind safe
COLOR_TRAIN = "#1f77b4"   # blue
COLOR_TEST  = "#d62728"   # red
COLOR_VAL   = "#2ca02c"   # green
COLOR_GP    = "#9467bd"   # purple
COLOR_LR    = "#7f7f7f"   # grey
COLOR_TIAG  = "#1f77b4"   # Ti-Al-Cr blue
COLOR_TINIFE= "#d62728"   # Ti-Ni-Fe red

plt.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "font.size"         : 11,
    "axes.titlesize"    : 12,
    "axes.labelsize"    : 11,
    "legend.fontsize"   : 9,
    "xtick.labelsize"   : 10,
    "ytick.labelsize"   : 10,
    "figure.dpi"        : 150,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.grid"         : True,
    "grid.alpha"        : 0.3,
    "grid.linewidth"    : 0.5,
})


# 2. DATA LOADING AND FIXED SPLIT DEFINITION
print("=" * 70)
print("  PySR Symbolic Regression - Cryogenic SMA (Song 2026 primary data)")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
df["T_K"]   = pd.to_numeric(df["T_K"],   errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df.dropna(subset=["T_K", "value"], inplace=True)

print(f"\nLoaded {len(df)} rows across {df['alloy_system'].nunique()} alloy systems")

# Ti-Al-Cr sigma_Ms: TRAINING ALLOY
# These are the primary data points from Song 2026 Fig 1b (compression)
# The stress-temperature trend is the target for empirical symbolic regression.
tiAlCr_Ms = df[
    (df["alloy_system"] == "Ti-Al-Cr") &
    (df["property"].isin(["sigma_Ms", "upper_plateau"]))
].dropna(subset=["T_K", "value"]).sort_values("T_K").copy()

# FIXED split: same as M1 (GPR) and M2 (PINN) for fair comparison
# 80% train (dataset "train" rows), 20% test (dataset "test" rows)
tiAlCr_train = tiAlCr_Ms[tiAlCr_Ms["split"] == "train"].copy()
tiAlCr_test  = tiAlCr_Ms[tiAlCr_Ms["split"] == "test"].copy()

T_train = tiAlCr_train["T_K"].values.astype(float)
y_train = tiAlCr_train["value"].values.astype(float)
T_test  = tiAlCr_test["T_K"].values.astype(float)
y_test  = tiAlCr_test["value"].values.astype(float)

print(f"\nTi-Al-Cr σ_Ms - Training: n={len(T_train)}, Test: n={len(T_test)}")
print(f"  T range: [{T_train.min():.0f}, {T_train.max():.0f}] K (train) "
      f"[{T_test.min():.0f}, {T_test.max():.0f}] K (test)")

# VALIDATION SET: independent alloy family
# FROZEN - never touched during PySR training or hyperparameter selection

# Cu-Al-Mn: use Ms transformation temperatures, not stress
val_CuAlMn = df[
    (df["alloy_system"] == "Cu-Al-Mn") &
    (df["property"] == "Ms") &
    (df["split"] == "validation")
].copy()

print(f"\nValidation sets (FROZEN - not used in any training):")
print(f"  Cu-Al-Mn Ms:     n={len(val_CuAlMn)}")

# Caloric data from Ti-Al-Cr (Song 2026)
tiAlCr_Qdiss  = df[(df["alloy_system"]=="Ti-Al-Cr") & (df["property"]=="Qdiss")
                   ].dropna(subset=["T_K","value"]).sort_values("T_K").copy()
tiAlCr_DSeff  = df[(df["alloy_system"]=="Ti-Al-Cr") & (df["property"]=="DeltaSeff")
                   ].dropna(subset=["T_K","value"]).sort_values("T_K").copy()
tiAlCr_dTad_meas = df[
    (df["alloy_system"]=="Ti-Al-Cr") & (df["property"]=="dTad_meas")
].dropna(subset=["T_K","value"]).sort_values("T_K").copy()

print(f"\n  Ti-Al-Cr Q_diss:    n={len(tiAlCr_Qdiss)}")
print(f"  Ti-Al-Cr ΔS_eff:    n={len(tiAlCr_DSeff)}")


# 3. BASELINE: LINEAR REGRESSION (Clausius-Clapeyron linear approximation)
# This is the model PySR must beat.
# Liu 2024b reports R²=0.92, MAE=23.4°C for ambient NiTi (gradient boosting).
# The linear CC model (sigma = a + b*T) is the simplest physics-justified baseline.
print("\nBaseline: Linear CC Regression")

lr_baseline = LinearRegression()
lr_baseline.fit(T_train.reshape(-1, 1), y_train)

y_train_lr = lr_baseline.predict(T_train.reshape(-1, 1))
y_test_lr  = lr_baseline.predict(T_test.reshape(-1, 1))

lr_r2_train = r2_score(y_train, y_train_lr)
lr_r2_test  = r2_score(y_test,  y_test_lr)
lr_mae_test = mean_absolute_error(y_test, y_test_lr)
lr_rmse_test= np.sqrt(mean_squared_error(y_test, y_test_lr))

print(f"  σ_c = {lr_baseline.coef_[0]:.4f}·T + {lr_baseline.intercept_:.2f}")
print(f"  Train R² = {lr_r2_train:.4f}")
print(f"  Test  R² = {lr_r2_test:.4f},  MAE = {lr_mae_test:.2f} MPa,  "
      f"RMSE = {lr_rmse_test:.2f} MPa")
print(f"  NOTE: linear CC fails at CC inversion - predicts monotone increase "
      f"while data inverts at ~{CC_INV_T:.0f}K")


# 4. SINGLE TEMPERATURE FEATURE
# Use a single dimensionless temperature feature to avoid redundant inputs.
# Use temperature alone to avoid redundant, deterministic transformations of T.

T_char  = 100.0   # feature scale in K
sig_char= 380.0   # reference stress (σ_Ms at 300K, Song 2026)

def engineer_features(T_K):
    """Return the single dimensionless temperature feature T/100 K."""
    return (np.atleast_1d(T_K).astype(float) / T_char).reshape(-1, 1)

feature_names = ["T100"]
X_train = engineer_features(T_train)
X_test  = engineer_features(T_test)

# Normalize target to [0,1] range for PySR numerical stability
# PySR works best when target values are O(1)
y_scaler   = StandardScaler()
y_train_sc = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

print(f"\nFeature engineering complete")
print(f"  Input features: {feature_names}")
print(f"  Target normalization: mean={y_scaler.mean_[0]:.2f}, "
      f"scale={y_scaler.scale_[0]:.2f} MPa")


# 5. PYSR SYMBOLIC REGRESSION - MODEL M3
print("\n" + "=" * 70)
print("  PySR SYMBOLIC REGRESSION - Model M3")
print("=" * 70)

def run_pysr(X_train, y_train_sc, feature_names):
    """
    Run PySR with publication-quality configuration.

    PySR implements multi-population evolutionary algorithm (Cranmer 2023).
    Key parameters justified below with literature references.

    Returns: PySRRegressor fitted object
    """
    from pysr import PySRRegressor

    model = PySRRegressor(
        # SEARCH SPACE
        # Operators: physics-motivated set following Sahoo 2026 precedent
        # for symbolic regression on SMA datasets
        binary_operators=["+", "-", "*"],
        unary_operators=["exp", "square"],

        # COMPLEXITY CONTROL
        # nterms_per_side controls max expression tree size
        # 15 nodes allows: sigma = a + b*T + c*exp(-T/d) type expressions
        # sufficient to capture anomalous inversion (Song 2026)
        maxsize             = 12,  # max tree nodes
        maxdepth            = 7,   # max tree depth

        # Parsimony: penalizes complexity on Pareto front
        # 0.001 is conservative - prioritizes accuracy slightly over simplicity
        # Increase to 0.01 if discovered equations are too complex
        parsimony           = 0.001,

        # EVOLUTIONARY ALGORITHM
        # niterations: number of generations
        # Each generation: population undergoes mutation, crossover, selection

           # minimum for physically meaningful discovery
        niterations         = PYSR_NITERATIONS,
        # Population size: number of candidate equations per island
        # 50 is PySR default and appropriate for N=10 training points
        populations         = 30,    # number of islands (parallel populations)
        population_size     = 50,    # individuals per island

        # Tournament selection: 10 is PySR default, good for sparse data
        tournament_selection_n = 10,

        # LOSS FUNCTION
        # Mean squared error on normalized target (standard for regression)
        # MSE chosen over MAE: MSE penalizes large residuals at CC inversion
        # region more heavily, pushing PySR to capture the sign change
        elementwise_loss    = "loss(prediction, target) = (prediction - target)^2",

        # PHYSICAL CONSTRAINTS
        # Constrain coefficient search to physically reasonable ranges
        # All in normalized units (y is normalized by y_scaler)
        # Prevents: sigma < 0 at T > 0, sigma > 10 GPa (unphysical)
        constraints         = {"exp": 9},

        # REPRODUCIBILITY
        random_state        = 42,
        parallelism = "serial", deterministic = True,
        # OUTPUT CONTROL
        verbosity           = 1,      # print progress
        progress            = True,   # progress bar
        print_precision     = 5,      # decimal places in printed equations

        # MULTITHREADING
        # Use all available CPU cores for evolutionary search
        # Set to 1 if running in restricted environment

        # FEATURE SELECTION
        # Allow PySR to select which features to use
        # All 6 physics features are candidates
        select_k_features   = None,  # None = use all features

        # EQUATION FILE
        # Save intermediate equations to disk (allows resuming)
        temp_equation_file  = True,

        # BATCHING
        # With N=10 training points, use full batch (no batching)
        batch_size          = 10,
        batching            = False,
    )

    return model


# 6. TRAINING AND EQUATION EXTRACTION
print(f"\nStarting PySR with {PYSR_NITERATIONS} iterations...")
print(f"Training data: n={len(X_train)}, features={feature_names}")
print(f"Normalized target: mean={y_scaler.mean_[0]:.1f}, scale={y_scaler.scale_[0]:.1f} MPa")
print(f"\nNote: First run installs Julia (~5-10 min). Subsequent runs: ~{PYSR_NITERATIONS//10}-{PYSR_NITERATIONS//5} min")

def sigma_form_A_plus_BT_plus_C_exp(T, A, B, C, T0, w):
    return A + B * T + C / (1.0 + np.exp((T - T0) / w))

t_start = time.time()
model = None
pysr_expression = None
best_loo_r2 = float("nan")
_saved_model_path = f"{OUT_DIR}/pysr_tiAlCr_model.pkl"

try:
    if os.path.exists(_saved_model_path):
        print(f"  Loading saved model from {_saved_model_path}")
        with open(_saved_model_path, "rb") as _f:
            model = pickle.load(_f)
        try:
            model.predict(X_train[:1])
            print("  Saved model feature shape is compatible.")
        except (ValueError, IndexError):
            print("  Saved model uses a different feature shape; retraining.")
            model = None
    if model is None:
        model = run_pysr(X_train, y_train_sc, feature_names)
        model.fit(X_train, y_train_sc)

    t_elapsed = time.time() - t_start
    print(f"\nPySR completed in {t_elapsed:.1f} s ({t_elapsed/60:.1f} min)")

    # Extract Pareto front
    pareto = model.equations_.copy()
    pareto["loss_sqrt"] = np.sqrt(pareto["loss"])  # RMSE in normalized units

    # Denormalize loss to MPa for physical interpretation
    pareto["RMSE_MPa"] = pareto["loss_sqrt"] * y_scaler.scale_[0]
    print(f"\nPareto front: {len(pareto)} equations")
    print(f"\nTop 5 equations by accuracy:")
    print(pareto[["complexity","loss","RMSE_MPa","equation"]].head(10).to_string())

    # Select by leave-one-out R² after refitting equation constants in each fold.
    import sympy as sp
    x0 = sp.Symbol("x0")

    def loo_r2_for_equation(eq_index):
        expr = model.sympy(int(eq_index))
        constants = sorted(expr.atoms(sp.Float), key=float)
        constant_symbols = sp.symbols(f"c0:{len(constants)}")
        expr_fit = expr.xreplace(dict(zip(constants, constant_symbols)))
        predictor = sp.lambdify((x0, *constant_symbols), expr_fit, "numpy")
        oof = np.full(len(X_train), np.nan)
        for fold_train, fold_test in LeaveOneOut().split(X_train):
            x_fold = X_train[fold_train, 0]
            y_fold = y_train_sc[fold_train]
            initial = np.asarray([float(value) for value in constants], dtype=float)

            def residual(params):
                with np.errstate(all="ignore"):
                    predicted = np.asarray(predictor(x_fold, *params), dtype=float)
                return np.nan_to_num(predicted - y_fold, nan=1e6,
                                     posinf=1e6, neginf=-1e6)

            if len(initial):
                fit = least_squares(residual, initial, max_nfev=300)
                params = fit.x
            else:
                params = ()
            with np.errstate(all="ignore"):
                oof[fold_test] = np.asarray(
                    predictor(X_train[fold_test, 0], *params), dtype=float).reshape(-1)
        return float(r2_score(y_train_sc, oof))

    pareto_sorted = pareto.sort_values("complexity").copy()
    pareto_sorted["score"] = [
        loo_r2_for_equation(eq_index) for eq_index in pareto_sorted.index
    ]
    best_eq_row = pareto_sorted.loc[pareto_sorted["score"].idxmax()]
    best_loo_r2 = float(best_eq_row["score"])
    pareto_sorted.to_csv(f"{OUT_DIR}/pysr_equations_pareto.csv", index=True)
    best_eq_str  = str(best_eq_row["equation"])
    best_eq_index = int(best_eq_row.name)
    best_complexity = int(best_eq_row["complexity"])
    best_rmse_mpa   = float(best_eq_row["RMSE_MPa"])
    pysr_expression = model.sympy(best_eq_index)

    print(f"\nBest equation (maximum LOO R², complexity={best_complexity})")
    print(f"  {best_eq_str}")
    print(f"  In-sample RMSE = {best_rmse_mpa:.2f} MPa; "
          f"LOO R² = {best_eq_row['score']:.4f}")

    # Get predictions from best model
    # model.predict returns normalized values - must inverse transform
    y_pred_train_sc = model.predict(X_train, index=best_eq_index)
    y_pred_test_sc  = model.predict(X_test, index=best_eq_index)

    y_pred_train = y_scaler.inverse_transform(y_pred_train_sc.reshape(-1, 1)).ravel()
    y_pred_test  = y_scaler.inverse_transform(y_pred_test_sc.reshape(-1, 1)).ravel()

    # Save model
    with open(f"{OUT_DIR}/pysr_tiAlCr_model.pkl", "wb") as f:
        pickle.dump(model, f)
        print(f"  PKL saved - delete this file to rerun PySR, keep it to load saved model")

except Exception as _pysr_err:
    import traceback; traceback.print_exc()
    if os.environ.get("ALLOW_PYSR_FALLBACK") != "1":
        raise
    print(f"\nPySR unavailable ({type(_pysr_err).__name__}: {str(_pysr_err)[:120]})")
    print("Generating an illustrative empirical curve fit instead.")

    # The fallback is an illustrative curve fit, not a PySR discovery.


    # Global optimization with differential evolution (avoids local minima)
    # Pure curve_fit fails here because the loss landscape has multiple basins
    # Reference: Storn & Price 1997, differential evolution for nonlinear optimization
    from scipy.optimize import differential_evolution

    def _cost_sigmoid(params):
        return np.sum((sigma_form_A_plus_BT_plus_C_exp(T_train, *params) - y_train)**2)

    _bounds_sig = [(-300., 500.), (0., 5.), (0., 2000.), (30., 300.), (5., 150.)]
    _result_sig = differential_evolution(_cost_sigmoid, _bounds_sig, seed=42,
                                         maxiter=20000, tol=1e-14, polish=True,
                                         popsize=30)
    popt, pcov = _result_sig.x, None
    # pcov from differential_evolution is unavailable; use curve_fit for uncertainties
    try:
        popt, pcov = curve_fit(sigma_form_A_plus_BT_plus_C_exp, T_train, y_train,
                               p0=popt, maxfev=50000)
    except Exception:
        pcov = np.diag(np.ones(5) * 1e6)  # fallback: large uncertainty

    A_fit, B_fit, C_fit, T0_fit, w_fit = popt
    tau_fit = w_fit  # keep tau_fit name for downstream compatibility

    if pcov is not None and not np.any(np.isinf(np.diag(pcov))):
        perr = np.sqrt(np.diag(pcov))
    else:
        perr = np.ones(5) * float("nan")

    print(f"  Fitted form: σ(T) = {A_fit:.2f} + {B_fit:.4f}·T + "
          f"({C_fit:.2f})/(1+exp((T-{T0_fit:.2f})/{w_fit:.2f}))")
    print(f"  Parameter uncertainties: ΔA={perr[0]:.2f}, ΔB={perr[1]:.4f}, "
          f"ΔC={perr[2]:.2f}, ΔT0={perr[3]:.2f}, Δw={perr[4]:.2f}")

    # Compute predictions on train/test sets using fitted parameters
    y_pred_train = sigma_form_A_plus_BT_plus_C_exp(T_train, A_fit, B_fit, C_fit, T0_fit, w_fit)
    y_pred_test = sigma_form_A_plus_BT_plus_C_exp(T_test, A_fit, B_fit, C_fit, T0_fit, w_fit)

    # Format equation string for display
    best_eq_str = (f"{A_fit:.2f} + {B_fit:.4f}·T + "
                   f"({C_fit:.2f})/(1+exp((T-{T0_fit:.2f})/{w_fit:.2f}))")
    best_complexity = 9  # Complexity of sigmoid form in nodes
    best_rmse_mpa = float(np.sqrt(mean_squared_error(y_train, y_pred_train)))

    # Create a minimal Pareto dataframe for consistency with PySR output
    rmse_train_norm = mean_absolute_error(y_train, y_pred_train) / y_scaler.scale_[0]
    rmse_test_norm = mean_absolute_error(y_test, y_pred_test) / y_scaler.scale_[0]
    pareto = pd.DataFrame({
        "complexity": [best_complexity],
        "loss": [rmse_train_norm],
        "loss_sqrt": [np.sqrt(rmse_train_norm)],
        "RMSE_MPa": [rmse_train_norm * y_scaler.scale_[0]],
        "equation": [best_eq_str],
        "r2_train": [r2_score(y_train, y_pred_train)],
        "r2_test": [r2_score(y_test, y_pred_test)]
    })

# 7. METRICS ON TRAINING AND TEST SET
print("\nPySR M3 Metrics")

r2_train  = r2_score(y_train, y_pred_train)
r2_test   = r2_score(y_test,  y_pred_test)
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test  = mean_absolute_error(y_test,  y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print(f"  Training:   R² = {r2_train:.4f},  MAE = {mae_train:.2f} MPa")
print(f"  Test:       R² = {r2_test:.4f},  MAE = {mae_test:.2f} MPa,  "
      f"RMSE = {rmse_test:.2f} MPa")
print(f"  Equation:   {best_eq_str}")
print(f"  Complexity: {best_complexity} nodes")
print(f"  Baseline (linear CC): R²={lr_r2_test:.4f}, MAE={lr_mae_test:.2f} MPa")
if r2_test > lr_r2_test:
    improvement = (r2_test - lr_r2_test) / (1 - lr_r2_test) * 100
    print(f"  Improvement over linear baseline: {improvement:.1f}% in unexplained variance")
    # Data leakage check: confirm test points not in training set
    overlap = set(T_test.tolist()) & set(T_train.tolist())
    print(f"  Train/test overlap: {len(overlap)} points (must be 0)")
    assert len(overlap) == 0, "DATA LEAKAGE DETECTED"


# 8. CC INVERSION DETECTION FROM SELECTED EMPIRICAL EQUATION
# Examine whether the selected empirical equation has a dσ/dT sign change.
# Song 2026 reports compression inversion at ~150K.
# Tolerance ±25K accepted (consistent with digitization uncertainty).

print("\nCC Inversion Detection")
print("  Reference: Song 2026 - compression inversion at ~150K")
# Always fit the physical form regardless of PySR outcome
# This is used for curve evaluation and bootstrap.
try:
    popt_phys, pcov_phys = curve_fit(
        sigma_form_A_plus_BT_plus_C_exp,
        T_train, y_train,
        p0    = [0., 1.4, 720., 150., 60.],
        bounds= ([0., 0., 0., 30., 5.], [500., 10., 2000., 300., 150.]),
        maxfev= 50000
    )
    A_fit, B_fit, C_fit, T0_fit, w_fit = popt_phys
    tau_fit = w_fit
    print(f"  Physical form fit: σ(T) = {A_fit:.2f} + {B_fit:.4f}·T + ({C_fit:.2f})·exp(-T/{tau_fit:.2f})")
except RuntimeError:
    A_fit, B_fit, C_fit, T0_fit, w_fit = 0., 1.5, 500., 150., 60.
    tau_fit = w_fit
    print("  Physical form fit failed - using default parameters")

_A, _B, _C, _tau = A_fit, B_fit, C_fit, tau_fit
_T0, _w = T0_fit, w_fit
T_dense = np.linspace(5., T_MAX_PHYSICAL, 2000)
X_dense = engineer_features(T_dense)

# Evaluate the selected PySR equation in physical units. All constants are
# read from this run's selected symbolic expression, never from stale literals.
_scale_y  = float(y_scaler.scale_[0])
_mean_y   = float(y_scaler.mean_[0])

if pysr_expression is not None:
    x0 = sp.Symbol("x0")
    sigma_normalized = sp.lambdify(x0, pysr_expression, "numpy")
    dsigma_dx = sp.lambdify(x0, sp.diff(pysr_expression, x0), "numpy")

    def sigma_pysr(T):
        return sigma_normalized(np.asarray(T) / T_char) * _scale_y + _mean_y

    def dsigma_pysr_dT(T):
        return dsigma_dx(np.asarray(T) / T_char) / T_char * _scale_y
else:
    def sigma_pysr(T):
        return sigma_form_A_plus_BT_plus_C_exp(
            np.asarray(T), A_fit, B_fit, C_fit, T0_fit, w_fit)

    def dsigma_pysr_dT(T):
        z = (np.asarray(T) - T0_fit) / w_fit
        ez = np.exp(z)
        return B_fit - (C_fit / w_fit) * ez / (1. + ez) ** 2

y_dense = sigma_pysr(T_dense)
# Numerical gradient: dσ/dT
dSigma_dT = np.gradient(y_dense, T_dense)

# Find sign changes (zero crossings in gradient)
sign_changes = np.where(np.diff(np.sign(dSigma_dT)))[0]
inversion_Ts = []
for idx in sign_changes:
    T_inv = 0.5 * (T_dense[idx] + T_dense[idx+1])
    inversion_Ts.append(T_inv)

# Filter: only report sign changes within training temperature range.
inversion_Ts_physical = [t for t in inversion_Ts
                         if T_MIN_DATA <= t <= T_MAX_PHYSICAL]
print(f"  Found {len(inversion_Ts)} sign change(s) total; "
      f"{len(inversion_Ts_physical)} within physical domain "
      f"({T_MIN_DATA:.0f}-{T_MAX_PHYSICAL:.0f}K):")
for T_inv in inversion_Ts_physical:
    err = abs(T_inv - CC_INV_T)
    status = "PASS" if err < 25. else ("MARGINAL" if err < 50. else "FAIL")
    print(f"    T_inv = {T_inv:.1f} K  |  Reference: {CC_INV_T:.0f} K  |  "
          f"Error: {err:.1f} K  |  [{status}]")
# Use physical inversions downstream
inversion_Ts = inversion_Ts_physical

if len(inversion_Ts) == 0:
    print(f"  WARNING: No sign inversion detected.")
    print(f"  This may indicate insufficient training iterations or data density.")
    print(f"  Recommendation: increase niterations to {PYSR_NITERATIONS*5} and retrain.")


# 9. CROSS-ALLOY TRANSFER STATUS
print("\nCross-Alloy Transfer")
transfer_results = []
print("  Not reported: Mg-Sc data require re-digitization; Ti-Ni-Fe needs a")
print("  constrained LOO analysis before any transfer score is reported.")


# 10. CALORIC CHAIN FROM SELECTED EMPIRICAL EQUATION
# Use the discovered σ(T) to derive ΔS(T), ΔS_eff(T), ΔTad(T), and COP_mat(T)
# via the Song 2026 thermodynamic chain:
#   ΔS(T)    = -ε_tr × (dσ/dT) × 10^6 / ρ           [CC relation, Eq.1]
#   ΔSeff(T) = ΔS(T) + Qdiss(T) / (2T)               [Song Eq.2]
#   ΔTad is taken from direct measurements. A constant-Cp comparison is shown
#   only for T >= 200 K and checked against the measured value at 294 K.
#   COP_mat  = Cp × |ΔTad| / Qdiss                    [Song Eq.4]

print("\nCaloric Chain from Discovered Equation")
print("  Reference: Song 2026 Eqs. 1-4")


# Caloric chain uses DIRECT measured ΔSeff and Qdiss datasets (Song 2026).
# The PySR sigma(T) equation contributes to CC inversion detection (Section 8).
# The caloric chain is a SEPARATE contribution using measured thermodynamic data.
# This is the scientifically correct separation: σ(T) model ≠ caloric model.
#
# Using model-derived ΔS_CC from dσ/dT gives zero crossing at ~129K because
# the exponential fit underestimates the high-T CC slope magnitude.
# Direct ΔSeff from Song 2026 dataset gives zero crossing at ~100K (ref: 97K).
# Reference: Song 2026 Fig 4b - ΔSeff measured directly from calorimetry.

from scipy.interpolate import interp1d

# Direct Qdiss from dataset (cubic spline, constant extrapolation at boundaries)
T_Qd = tiAlCr_Qdiss["T_K"].values
y_Qd = tiAlCr_Qdiss["value"].values
Q_interp = interp1d(T_Qd, y_Qd, kind="cubic",
                    fill_value=(y_Qd[0], y_Qd[-1]),  # constant, not linear extrap
                    bounds_error=False)
Q_dense = np.maximum(Q_interp(T_dense), 0.)

# Direct ΔSeff from dataset (Song 2026 values)
T_DSeff_d = tiAlCr_DSeff["T_K"].values
y_DSeff_d = tiAlCr_DSeff["value"].values
DSeff_interp = interp1d(T_DSeff_d, y_DSeff_d, kind="cubic",
                        fill_value="extrapolate", bounds_error=False)
DeltaSeff_chain = DSeff_interp(T_dense)

# Keep the Cp-based comparison out of the cryogenic regime.
DeltaTad_chain = np.full_like(T_dense, np.nan)
high_temp_mask = T_dense >= 200.
DeltaTad_chain[high_temp_mask] = (
    T_dense[high_temp_mask] / CP) * DeltaSeff_chain[high_temp_mask]

# Also retain model-derived ΔS for the CC comparison plot in Figure 3
dSigma_dT_MPaK = np.gradient(y_dense, T_dense)          # MPa/K (for plot only)
DeltaS_CC       = -EPS_TR * dSigma_dT_MPaK * 1e6 / RHO  # J/kg·K (for plot only)

mask_cooling = (DeltaSeff_chain < 0) & (T_dense >= 50.) & (T_dense <= T_MAX_PHYSICAL)
COP_chain    = np.full_like(T_dense, np.nan)
COP_chain[mask_cooling] = (T_dense[mask_cooling] * np.abs(DeltaSeff_chain[mask_cooling])
                           / Q_dense[mask_cooling])

T_val_pt = 294.
dTad_measured_294 = float(np.interp(
    T_val_pt, tiAlCr_dTad_meas["T_K"], tiAlCr_dTad_meas["value"]))
dTad_check_294 = float(np.interp(T_val_pt, T_dense, DeltaTad_chain))
DeltaSeff_294 = float(np.interp(T_val_pt, T_dense, DeltaSeff_chain))
COP_297 = float(np.interp(
    297., T_dense, np.where(np.isnan(COP_chain), 0, COP_chain)))

print(f"\n  Caloric data at T=294K:")
print(f"    ΔS_eff   = {DeltaSeff_294:.2f} J/kg·K")
print(f"    ΔTad     = {dTad_measured_294:.2f} K (direct measurement)")
print(f"    ΔSeff/Cp = {dTad_check_294:.2f} K (consistency check only)")
print(f"    COP_mat at 297K = {COP_297:.2f} (Song 2025: 4.6)")

# Zero crossing of ΔSeff (lower operational limit for cooling)
dseff_sign_changes = np.where(np.diff(np.sign(DeltaSeff_chain)))[0]
T_zero_eff = []
for idx in dseff_sign_changes:
    T_zero_eff.append(0.5*(T_dense[idx]+T_dense[idx+1]))
if T_zero_eff:
    print(f"    ΔSeff zero crossing: T ≈ {T_zero_eff[0]:.0f} K "
          f"(Song 2026 reports ~97K)")


# 11. INDEPENDENT CALORIMETRIC CLAUSIUS-CLAPEYRON TEST
print("\nIndependent calorimetric CC test against measured σ_0")
print("  Reconstruct ΔS = ΔS_eff - Q_diss/(2T) from measured Song caloric data.")
print("  Integrate dσ_0/dT = -ΔS·ρ/ε_tr, anchored at measured σ_0(300K).")

sigma0_data = df[
    (df["alloy_system"] == "Ti-Al-Cr") & (df["property"] == "sigma_0")
].sort_values("T_K")
T_sigma0 = sigma0_data["T_K"].to_numpy(dtype=float)
y_sigma0 = sigma0_data["value"].to_numpy(dtype=float)
anchor_idx = int(np.argmin(np.abs(T_sigma0 - 300.)))
T_anchor = float(T_sigma0[anchor_idx])
sigma0_anchor = float(y_sigma0[anchor_idx])

T_cc_min = max(50., float(T_Qd.min()), float(T_DSeff_d.min()))
T_cc_max = min(T_MAX_PHYSICAL, float(T_Qd.max()), float(T_DSeff_d.max()))
T_cc_grid = np.unique(np.append(np.linspace(T_cc_min, T_cc_max, 1200), T_anchor))
Q_cc = np.interp(T_cc_grid, T_dense, Q_dense)
Seff_cc = np.interp(T_cc_grid, T_dense, DeltaSeff_chain)
dS = df[(df.alloy_system=="Ti-Al-Cr")&(df.property=="DeltaS_cal")].sort_values("T_K")
S_fn = interp1d(dS.T_K, dS.value, kind="cubic", bounds_error=False,
                fill_value=(dS.value.iloc[0], dS.value.iloc[-1]))
DeltaS_cal = S_fn(T_cc_grid)
dsigma0_dT_cal = -DeltaS_cal * RHO / EPS_TR / 1e6
sigma0_integral = cumulative_trapezoid(dsigma0_dT_cal, T_cc_grid, initial=0.)
sigma0_integrated = (sigma0_anchor + sigma0_integral
                     - np.interp(T_anchor, T_cc_grid, sigma0_integral))

compare_mask = ((T_sigma0 >= T_cc_min) & (T_sigma0 <= T_cc_max)
                & (np.arange(len(T_sigma0)) != anchor_idx))
T_cc_compare = T_sigma0[compare_mask]
y_sigma0_compare = y_sigma0[compare_mask]
sigma0_cc_compare = np.interp(T_cc_compare, T_cc_grid, sigma0_integrated)
cc_sigma_rmse = float(np.sqrt(mean_squared_error(y_sigma0_compare, sigma0_cc_compare)))
cc_sigma_mae = float(mean_absolute_error(y_sigma0_compare, sigma0_cc_compare))
cc_sigma_r2 = float(r2_score(y_sigma0_compare, sigma0_cc_compare))
print(f"  Independent σ_0 comparisons (anchor excluded): n={len(T_cc_compare)}")
print(f"  CC integration RMSE={cc_sigma_rmse:.2f} MPa, "
      f"MAE={cc_sigma_mae:.2f} MPa, R²={cc_sigma_r2:.4f}")


# 12. UNCERTAINTY QUANTIFICATION (Parametric Bootstrap)
# For PySR models, true uncertainty comes from:
# (a) The residual scatter of the discovered equation (aleatoric)
# (b) Sensitivity of the discovered form to training data perturbations (epistemic)
# Method: parametric bootstrap - perturb training data by digitization noise,
# re-fit coefficients N_boot times, compute CI from coefficient distribution.
# Reference: standard bootstrap protocol (Efron & Tibshirani 1993)

print("\nUncertainty Quantification (Parametric Bootstrap)")
print("  Method: N=200 bootstrap resamples of training data")
print("  Digitization noise: σ_stress=±2 MPa, σ_T=±3K (WebPlotDigitizer estimate)")

SIGMA_STRESS_NOISE = 2.   # MPa  - digitization uncertainty (WebPlotDigitizer)
SIGMA_T_NOISE      = 3.   # K    - temperature uncertainty
N_BOOT             = 200

# Bootstrapped predictions at each T in dense grid
y_boot_array = np.zeros((N_BOOT, len(T_dense)))

# Compute training residual std for prediction interval
y_train_nominal = sigma_form_A_plus_BT_plus_C_exp(T_train, A_fit, B_fit, C_fit, T0_fit, w_fit)
resid_std = float(np.std(y_train - y_train_nominal))

for b in range(N_BOOT):
    rng = np.random.RandomState(42 + b)
    T_b = T_train + rng.normal(0., SIGMA_T_NOISE,     size=len(T_train))
    y_b = y_train + rng.normal(0., SIGMA_STRESS_NOISE, size=len(y_train))

    try:
        popt_b, _ = curve_fit(
            sigma_form_A_plus_BT_plus_C_exp,
            T_b, y_b,
            p0     = [A_fit, B_fit, C_fit, T0_fit, w_fit],
            bounds = ([-300., 0., 0., 30., 5.], [500., 5., 2000., 300., 150.]),
            maxfev = 10000
        )
        # Prediction interval = parameter uncertainty + aleatoric residual noise
        y_boot_array[b] = (sigma_form_A_plus_BT_plus_C_exp(T_dense, *popt_b)
                           + rng.normal(0., resid_std, size=len(T_dense)))
    except Exception:
        y_boot_array[b] = y_dense + rng.normal(0., resid_std, size=len(T_dense))

if pysr_expression is not None:
    _consts = sorted(pysr_expression.atoms(sp.Float), key=float)
    _csyms = sp.symbols(f"c0:{len(_consts)}")
    _pred_b = sp.lambdify((x0, *_csyms),
                          pysr_expression.xreplace(dict(zip(_consts, _csyms))), "numpy")
    _p0_b = np.array([float(c) for c in _consts], dtype=float)
    resid_std = float(np.std(y_train - y_pred_train))
    for b in range(N_BOOT):
        rng = np.random.RandomState(42 + b)
        T_b = T_train + rng.normal(0., SIGMA_T_NOISE, size=len(T_train))
        y_b = (y_train + rng.normal(0., SIGMA_STRESS_NOISE, size=len(y_train)) - _mean_y) / _scale_y
        def _res_b(p, T_b=T_b, y_b=y_b):
            with np.errstate(all="ignore"):
                v = np.broadcast_to(np.asarray(_pred_b(T_b / T_char, *p), dtype=float), y_b.shape)
            return np.nan_to_num(v - y_b, nan=1e6, posinf=1e6, neginf=-1e6)
        try:
            p_b = least_squares(_res_b, _p0_b, max_nfev=300).x if len(_p0_b) else _p0_b
            with np.errstate(all="ignore"):
                curve_b = np.broadcast_to(np.asarray(_pred_b(T_dense / T_char, *p_b), dtype=float),
                                          T_dense.shape) * _scale_y + _mean_y
        except Exception:
            curve_b = y_dense.copy()
        y_boot_array[b] = curve_b + rng.normal(0., resid_std, size=len(T_dense))
y_boot_mean  = y_boot_array.mean(axis=0)
y_boot_std   = y_boot_array.std(axis=0)
y_boot_ci95_lo = np.percentile(y_boot_array, 2.5,  axis=0)
y_boot_ci95_hi = np.percentile(y_boot_array, 97.5, axis=0)

# CI coverage on test set
T_test_covered = np.array([
    y_boot_ci95_lo[np.argmin(np.abs(T_dense - t))] <= y_t <=
    y_boot_ci95_hi[np.argmin(np.abs(T_dense - t))]
    for t, y_t in zip(T_test, y_test)
])
ci95_coverage = T_test_covered.mean() * 100
print(f"  Bootstrap 95% prediction interval coverage on test set: {ci95_coverage:.0f}% (expected >=90%)")
print(f"  (Prediction interval = parameter uncertainty + residual std={resid_std:.1f} MPa)")

# Uncertainty grows below 100K (CRYO-CRITICAL) as expected
below100K = T_dense < 100.
above100K = T_dense >= 100.
std_below100 = y_boot_std[below100K].mean() if below100K.any() else float("nan")
std_above100 = y_boot_std[above100K].mean() if above100K.any() else float("nan")
print(f"  Mean bootstrap std below 100K: {std_below100:.1f} MPa (expected > above)")
print(f"  Mean bootstrap std above 100K: {std_above100:.1f} MPa")
if std_below100 > std_above100:
    print(f"  STATUS: PASS - uncertainty correctly higher in cryogenic regime")
else:
    print(f"  STATUS: NOTE - uncertainty not markedly higher below 100K")
    print(f"  This may indicate limited sensitivity to the inversion in this run")


print("\nGenerating figures")

def shade_cryo(ax, ymin=None, ymax=None):
    """Add cryogenic temperature region shading."""
    ylim = ax.get_ylim()
    y_lo = ymin if ymin is not None else ylim[0]
    y_hi = ymax if ymax is not None else ylim[1]
    ax.axvspan(0.,  50.,  alpha=0.14, color="#d32f2f", zorder=0)
    ax.axvspan(50., 150., alpha=0.08, color="#ff9800", zorder=0)
    ax.set_ylim(y_lo, y_hi)


# FIGURE 1: Pareto Front
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Figure M3-1 - PySR Pareto Front: Accuracy vs. Complexity\n"
             "Ti-Al-Cr σ_Ms(T), Song 2026 data",
             fontsize=13, fontweight="bold")

ax = axes[0]
pareto_sorted_plot = pareto.sort_values("complexity")
ax.scatter(pareto_sorted_plot["complexity"],
           pareto_sorted_plot["RMSE_MPa"],
           c=COLOR_TIAG, s=80, zorder=5, edgecolors="white", lw=0.5,
           label="Pareto equations")
ax.plot(pareto_sorted_plot["complexity"],
        pareto_sorted_plot["RMSE_MPa"],
        "-", color=COLOR_TIAG, alpha=0.5, lw=1)

# Mark selected equation
ax.scatter([best_complexity], [best_rmse_mpa],
           c="gold", s=200, marker="*", zorder=8, edgecolors="k", lw=0.8,
           label=f"Selected (knee): complexity={best_complexity}")

# Mark linear CC baseline
ax.axhline(lr_rmse_test, color=COLOR_LR, ls="--", lw=1.5,
           label=f"Linear CC baseline: {lr_rmse_test:.1f} MPa")

ax.set_xlabel("Equation Complexity (nodes)")
ax.set_ylabel("RMSE (MPa, physical units)")
ax.set_title("(a) Pareto front - complexity vs. accuracy")
ax.legend(fontsize=8)
ax.set_xlim(left=2)

ax = axes[1]
# Show top 8 equations as table
top8 = pareto.nsmallest(8, "RMSE_MPa")[["complexity","RMSE_MPa","equation"]]
ax.axis("off")
col_labels = ["Complexity", "RMSE (MPa)", "Equation"]
table_data = [[str(row["complexity"]),
               f"{row['RMSE_MPa']:.2f}",
               str(row["equation"])[:50]] for _, row in top8.iterrows()]
tbl = ax.table(
    cellText    = table_data,
    colLabels   = col_labels,
    cellLoc     = "left",
    loc         = "center",
    colWidths   = [0.12, 0.14, 0.74],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.5)
# Highlight selected equation row
matching_indices = top8.reset_index().index[top8.RMSE_MPa == best_rmse_mpa].tolist()
if matching_indices:
    for j in range(3):
        tbl[(matching_indices[0]+1, j)].set_facecolor("#FFF9C4")
ax.set_title("(b) Top-8 Pareto equations", fontsize=11)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_M3_pareto.png", bbox_inches="tight", dpi=150)
plt.close()
print(f"  [saved] fig_M3_pareto.png")


# FIGURE 2: Main result - fit + cross-alloy transfer (2 panels)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Figure M3-2 - PySR: Discovered Equation Fit and Cross-Alloy Transfer",
             fontsize=13, fontweight="bold")

ax = axes[0]
# Cryo shading
ax.fill_between([0., 50.],  0, 800, alpha=0.14, color="#d32f2f", zorder=0,
                label="<50K: EXTREME-CRYO")
ax.fill_between([50., 150.],0, 800, alpha=0.08, color="#ff9800", zorder=0,
                label="50-150K: CRYO-CRITICAL")

# Bootstrap CI band
ax.fill_between(T_dense, y_boot_ci95_lo, y_boot_ci95_hi,
                alpha=0.20, color=COLOR_TIAG, zorder=1, label="95% CI (bootstrap)")

# Discovered equation (mean)
ax.plot(T_dense, y_dense, "-", color=COLOR_TIAG, lw=2.2, zorder=4,
        label=f"PySR: {best_eq_str[:60]}…" if len(best_eq_str) > 60 else f"PySR: {best_eq_str}")

# Linear baseline
T_plot_lr = np.linspace(T_dense.min(), T_dense.max(), 300)
ax.plot(T_plot_lr, lr_baseline.predict(T_plot_lr.reshape(-1, 1)), "--",
        color=COLOR_LR, lw=1.5, alpha=0.8, label="Linear CC baseline (classical)")

# Training data
ax.scatter(T_train, y_train, c=COLOR_TRAIN, s=90, zorder=6, edgecolors="white", lw=0.5,
           label=f"Train (n={len(T_train)})")

# Test data
ax.scatter(T_test, y_test, c=COLOR_TEST, s=120, marker="*", zorder=7,
           edgecolors="k", lw=0.5,
           label=f"Test (n={len(T_test)})")

# CC inversion vertical line
for T_inv in inversion_Ts[:1]:
    ax.axvline(T_inv, color="purple", lw=2., ls="--", alpha=0.85, zorder=5)
    ax.text(T_inv + 5, ax.get_ylim()[1]*0.75,
            f"CC inversion\n{T_inv:.0f} K\n(Song 2026: ~150K)",
            fontsize=8, color="purple")

# Annotations
ax.text(0.05, 0.95,
        f"Test R²={r2_test:.4f}\nMAE={mae_test:.1f} MPa\nComplexity={best_complexity}",
        transform=ax.transAxes, fontsize=9, va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("σ_Ms (MPa)")
ax.set_title("(a) PySR fit on Ti-Al-Cr (Song 2026)")
ax.legend(fontsize=7, loc="lower right")
ax.set_xlim(0., 510.)

ax = axes[1]
ax.text(0.5, 0.5, "Cross-alloy transfer omitted\npending source-data digitization",
        transform=ax.transAxes, ha="center", va="center", fontsize=10)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("σ_c (MPa)")
ax.set_title("(b) Cross-alloy transfer (not evaluated)")

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_M3_fit_and_transfer.png", bbox_inches="tight", dpi=150)
plt.close()
print(f"  [saved] fig_M3_fit_and_transfer.png")


# FIGURE 3: CC Slope Analysis
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Figure M3-3 - CC Slope Analysis from Discovered Equation\n"
             "Anomalous inversion versus classical prediction",
             fontsize=13, fontweight="bold")

ax = axes[0]
# dσ/dT from discovered equation
ax.fill_between([0., 50.],  dSigma_dT_MPaK.min()*1.1, dSigma_dT_MPaK.max()*1.1,
                alpha=0.14, color="#d32f2f", zorder=0)
ax.fill_between([50., 150.],dSigma_dT_MPaK.min()*1.1, dSigma_dT_MPaK.max()*1.1,
                alpha=0.08, color="#ff9800", zorder=0)
ax.axhline(0, color="k", lw=1.2, ls="--", label="dσ/dT = 0")
ax.bar(T_dense[::5], dSigma_dT_MPaK[::5],
       width = (T_dense.max() - T_dense.min()) / len(T_dense) * 4.5,
       color = ["#d62728" if v < 0 else COLOR_TIAG for v in dSigma_dT_MPaK[::5]],
       edgecolor="none", alpha=0.7, label="dσ/dT (MPa/K)")
for T_inv in inversion_Ts[:1]:
    ax.axvline(T_inv, color="purple", lw=2., ls=":", alpha=0.9,
               label=f"Sign change: {T_inv:.0f} K")

# Classical CC prediction (constant slope from high-T fit)
T_HT = T_dense[T_dense > 200.]
slope_classical = float(np.polyfit(T_HT, np.interp(T_HT, T_dense, y_dense), 1)[0])
ax.axhline(slope_classical, color=COLOR_LR, lw=1.5, ls="-.",
           label=f"Classical CC: {slope_classical:.2f} MPa/K")
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("dσ_Ms/dT (MPa/K)")
ax.set_title("(a) Clausius-Clapeyron slope\n(blue=classical, red=inverted)")
ax.legend(fontsize=7)
ax.set_xlim(0., 510.)

ax = axes[1]
# Σ comparison: Song raw data CC slope annotations
T_compare = np.array([50., 100., 150., 200., 250., 300., 350., 400.])
slope_compare = np.interp(T_compare, T_dense, dSigma_dT_MPaK)

# Literature slopes for comparison
_s0d = df[(df.alloy_system == "Ti-Al-Cr") & (df.property == "sigma_0")].sort_values("T_K")
_lo = _s0d[_s0d.T_K < 150]; _hi = _s0d[_s0d.T_K > 200]
_SLOPE_BELOW = float(np.polyfit(_lo.T_K, _lo.value, 1)[0]) if len(_lo) >= 2 else float("nan")
_SLOPE_ABOVE = float(np.polyfit(_hi.T_K, _hi.value, 1)[0]) if len(_hi) >= 2 else float("nan")
literature_slopes = {
    "Ti-Al-Cr below 150K\n(Song 2026, compression)"  : _SLOPE_BELOW,  # anomalous negative
    "Ti-Al-Cr above 200K\n(Song 2026, compression)"  : _SLOPE_ABOVE,  # positive
    "Ti-Ni-Fe\n(Zeng 2026)"                           : +1.49, # positive, small
    "NiTi binary\n(Otsuka 2005)"                      : +6.0,  # large positive
}
alloy_colors = [COLOR_TIAG, COLOR_TIAG, COLOR_TINIFE, "k"]
x_pos = np.arange(len(literature_slopes))
bars = ax.bar(x_pos, list(literature_slopes.values()),
              color=alloy_colors, edgecolor="white", alpha=0.85)
ax.axhline(0, color="k", lw=1)
ax.set_xticks(x_pos)
ax.set_xticklabels(list(literature_slopes.keys()), fontsize=7.5, rotation=10, ha="right")
ax.set_ylabel("CC slope dσ/dT (MPa/K)")
ax.set_title("(b) CC slope comparison:\ndiscovered vs. literature")
# Add discovered slope bars
disc_below = float(np.mean(dSigma_dT_MPaK[T_dense < 150.]))
disc_above = float(np.mean(dSigma_dT_MPaK[T_dense > 200.]))
ax.bar([-0.3, -0.3], [disc_below, disc_above], color="gold",
       width=0.2, label="PySR discovered", edgecolor="k", lw=0.8)
ax.legend(fontsize=8)

ax = axes[2]
# Comparison of PySR slope with CC-formula-implied slope from ΔSeff GP
# CC says: dσ/dT = -ΔS·ρ/ε_tr
# Use experimental ΔSeff(T) from Song 2026 directly
T_DSeff = tiAlCr_DSeff["T_K"].values
y_DSeff = tiAlCr_DSeff["value"].values
DSeff_interp = interp1d(T_DSeff, y_DSeff, kind="cubic",
                        fill_value="extrapolate", bounds_error=False)
DeltaSeff_data = DSeff_interp(T_dense)
cc_slope_from_data = -DeltaSeff_data * RHO / EPS_TR / 1e6  # MPa/K

ax.axhline(0, color="k", lw=1, ls="--")
ax.plot(T_dense, dSigma_dT_MPaK, "-", color=COLOR_TIAG, lw=2.2,
        label="PySR: dσ/dT from equation")
ax.plot(T_dense, cc_slope_from_data, "-.", color="darkorange", lw=1.8,
        label="CC-implied: -ΔSeff·ρ/ε_tr (Song 2026)")

# CC residual
cc_residual = np.abs(dSigma_dT_MPaK - cc_slope_from_data)
ax_r = ax.twinx()
ax_r.plot(T_dense, cc_residual, ":", color="red", lw=1.2, alpha=0.7,
          label="|Residual| (right)")
ax_r.set_ylabel("|CC residual| (MPa/K)", color="red", fontsize=9)
ax_r.tick_params(axis="y", colors="red")

cc_res_mean = cc_residual[(T_dense>=200.)&(T_dense<=400.)].mean()
ax.text(0.65, 0.92,
        f"Classical regime\nCC residual:\n{cc_res_mean:.3f} MPa/K",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

ax.fill_between([0., 50.],  -3, 8, alpha=0.12, color="#d32f2f", zorder=0)
ax.fill_between([50., 150.], -3, 8, alpha=0.07, color="#ff9800", zorder=0)
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("dσ/dT (MPa/K)")
ax.set_title("(c) PySR slope vs. CC-predicted slope")
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax_r.get_legend_handles_labels()
ax.legend(lines1+lines2, labs1+labs2, fontsize=7, loc="upper left")
ax.set_xlim(0., 510.)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_M3_slope_analysis.png", bbox_inches="tight", dpi=150)
plt.close()
print(f"  [saved] fig_M3_slope_analysis.png")


# FIGURE 4: Caloric chain + COP
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)
fig.suptitle("Figure M3-4 - Caloric Chain and Independent CC Check\n"
             "ΔS -> ΔS_eff -> ΔTad -> COP_mat  (Song 2026 framework)",
             fontsize=13, fontweight="bold")

# 4a: ΔS from CC (discovered equation slope) vs Song 2026 direct measurement
ax = fig.add_subplot(gs[0, 0])
ax.fill_between([0.,50.],  -40,40, alpha=0.14, color="#d32f2f", zorder=0)
ax.fill_between([50.,150.],-40,40, alpha=0.08, color="#ff9800", zorder=0)
ax.axhline(0, color="k", lw=1, ls="--")
ax.plot(T_dense, DeltaS_CC, "-", color=COLOR_TIAG, lw=2.2,
        label="ΔS from PySR slope (CC Eq.1)")
ax.scatter(T_DSeff, y_DSeff, c=COLOR_TEST, s=50, zorder=6, edgecolors="w",
           label="ΔS_eff data (Song 2026)")
ax.axvline(CC_INV_T, color="purple", ls="--", lw=1.5, alpha=0.8,
           label=f"CC inversion ~{CC_INV_T:.0f}K")
ax.set_xlabel("T (K)"); ax.set_ylabel("ΔS (J/kg·K)")
ax.set_title("(a) ΔS from CC × PySR slope")
ax.legend(fontsize=7); ax.set_xlim(0.,510.)

# 4b: Q_diss from Song 2026 data
ax = fig.add_subplot(gs[0, 1])
ax.fill_between([0.,50.],  900,1600, alpha=0.14, color="#d32f2f", zorder=0)
ax.fill_between([50.,150.],900,1600, alpha=0.08, color="#ff9800", zorder=0)
T_Qd_plot = tiAlCr_Qdiss["T_K"].values
y_Qd_plot = tiAlCr_Qdiss["value"].values
sc_Qd = ax.scatter(T_Qd_plot, y_Qd_plot,
                   c=[COLOR_TRAIN if s=="train" else COLOR_TEST
                      for s in tiAlCr_Qdiss["split"]],
                   s=45, zorder=6, edgecolors="w", lw=0.4)
ax.plot(T_dense, Q_dense, "-", color="green", lw=2, alpha=0.8, label="Interpolated Q_diss")
ax.set_xlabel("T (K)"); ax.set_ylabel("Q_diss (J/kg)")
ax.set_title("(b) Dissipation energy Q_diss(T)\n(Song 2026 data)")
ax.legend(fontsize=7); ax.set_xlim(0.,510.)

# 4c: ΔS_eff (combined chain)
ax = fig.add_subplot(gs[0, 2])
ax.fill_between([0.,50.],  -30,25, alpha=0.14, color="#d32f2f", zorder=0)
ax.fill_between([50.,150.],-30,25, alpha=0.08, color="#ff9800", zorder=0)
ax.axhline(0, color="k", lw=1.2, ls="--", label="ΔSeff = 0 (cooling boundary)")
ax.fill_between(T_dense, DeltaSeff_chain, 0,
                where=DeltaSeff_chain < 0, alpha=0.25, color=COLOR_TIAG,
                label="Cooling regime (ΔSeff < 0)")
ax.fill_between(T_dense, DeltaSeff_chain, 0,
                where=DeltaSeff_chain > 0, alpha=0.15, color="red",
                label="Net dissipation (ΔSeff > 0)")
ax.plot(T_dense, DeltaSeff_chain, "-", color=COLOR_TIAG, lw=2.2, label="ΔSeff(T) chain")
ax.scatter(T_DSeff, y_DSeff, c=COLOR_TEST, s=50, zorder=7, edgecolors="w",
           label="Song 2026 data")
if T_zero_eff:
    ax.axvline(T_zero_eff[0], color="darkred", lw=1.5, ls=":",
               label=f"Zero: {T_zero_eff[0]:.0f}K")
ax.set_xlabel("T (K)"); ax.set_ylabel("ΔS_eff (J/kg·K)")
ax.set_title("(c) Effective entropy ΔS_eff(T)\n(Song Eq. 2)")
ax.legend(fontsize=6.5); ax.set_xlim(0.,510.)

# 4d: measured ΔTad and high-temperature Cp consistency check
ax = fig.add_subplot(gs[1, 0])
ax.axhline(0, color="k", lw=1, ls="--")
ax.plot(T_dense, DeltaTad_chain, "-", color="navy", lw=2.2,
        label="ΔSeff/Cp check (T >= 200 K)")
ax.scatter(tiAlCr_dTad_meas["T_K"], tiAlCr_dTad_meas["value"],
           color="black", s=24, zorder=5, label="Directly measured ΔTad")
ax.scatter([294.], [dTad_check_294], marker="D", s=70, color="magenta",
           zorder=7, label="294 K consistency check")
ax.set_xlabel("T (K)"); ax.set_ylabel("ΔTad (K)")
ax.set_title("(d) Direct ΔTad measurements and high-T consistency check")
ax.legend(fontsize=6.5); ax.set_xlim(0.,510.)

# 4e: COP_mat
ax = fig.add_subplot(gs[1, 1])
ax.fill_between([0.,150.], -1, 25, alpha=0.08, color="#ff9800", zorder=0)
T_cop_mask = T_dense[mask_cooling]
cop_mask_v  = COP_chain[mask_cooling]
ax.plot(T_cop_mask, cop_mask_v, "-", color="navy", lw=2.2, label="COP_mat (chain)")
ax.axhline(4.6, color="gold", lw=2, ls="--", label="Song 2025: COP=4.6 @ 297K")
ax.scatter([297], [COP_297], s=200, marker="*", c="gold", zorder=8,
           edgecolors="k", lw=0.8, label=f"Chain: {COP_297:.2f}")
if len(cop_mask_v) > 0:
    best_cop_T = T_cop_mask[np.argmax(cop_mask_v)]
    best_cop_V = cop_mask_v.max()
    ax.axvline(best_cop_T, color="green", ls=":", lw=1.5, alpha=0.8,
               label=f"Max COP={best_cop_V:.1f} @ {best_cop_T:.0f}K")
ax.set_xlabel("T (K)"); ax.set_ylabel("COP_mat")
ax.set_title(f"(e) Coefficient of performance\n(Song Eq. 4, cooling regime only)")
ax.legend(fontsize=7); ax.set_xlim(50., 460.); ax.set_ylim(0., None)

# 4f: independent calorimetric CC integration against measured σ_0
ax = fig.add_subplot(gs[1, 2])
ax.plot(T_cc_grid, sigma0_integrated, "--", color="darkorange", lw=2,
        label="Integrated calorimetric CC prediction")
ax.scatter(T_sigma0, y_sigma0, c=COLOR_TEST, s=65, zorder=7,
           edgecolors="white", lw=0.5, label="Measured σ_0")
ax.scatter([T_anchor], [sigma0_anchor], c="black", marker="D", s=55,
           zorder=8, label=f"Measured anchor at {T_anchor:.0f}K")
ax.text(0.04, 0.96,
        f"Non-anchor RMSE={cc_sigma_rmse:.1f} MPa\n"
        f"MAE={cc_sigma_mae:.1f} MPa",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))
ax.set_xlabel("T (K)"); ax.set_ylabel("σ_0 (MPa)")
ax.set_title("(f) Independent calorimetric CC test\n"
             "integrated ΔS vs measured σ_0")
ax.legend(fontsize=7); ax.set_xlim(T_cc_min - 10, T_cc_max + 10)

plt.savefig(f"{OUT_DIR}/fig_M3_caloric_chain.png", bbox_inches="tight", dpi=150)
plt.close()
print(f"  [saved] fig_M3_caloric_chain.png")


# FIGURE 5: Uncertainty
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Figure M3-5 - Uncertainty Quantification (Parametric Bootstrap, N=200)\n"
             "Epistemic uncertainty grows in cryogenic regime as expected",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.fill_between([0.,50.],  0, 900, alpha=0.14, color="#d32f2f", zorder=0)
ax.fill_between([50.,150.],0, 900, alpha=0.08, color="#ff9800", zorder=0)
ax.fill_between(T_dense, y_boot_ci95_lo, y_boot_ci95_hi,
                alpha=0.25, color=COLOR_TIAG, label="95% CI (bootstrap)")
ax.plot(T_dense, y_dense,  "-",  color=COLOR_TIAG, lw=2.2, label="PySR mean prediction")
ax.plot(T_dense, y_boot_mean, "--", color="navy", lw=1.5, alpha=0.8,
        label="Bootstrap mean")
ax.scatter(T_train, y_train, c=COLOR_TRAIN, s=70, zorder=6, edgecolors="w")
ax.scatter(T_test,  y_test,  c=COLOR_TEST,  s=90, marker="*", zorder=7, edgecolors="k")
ax.set_xlabel("T (K)"); ax.set_ylabel("σ_Ms (MPa)")
ax.set_title(f"(a) Prediction with 95% bootstrap CI\n"
             f"CI coverage on test: {ci95_coverage:.0f}%")
ax.legend(fontsize=7); ax.set_xlim(0., 510.)

ax = axes[1]
ax.fill_between([0.,50.],  0, 60, alpha=0.14, color="#d32f2f", zorder=0)
ax.fill_between([50.,150.],0, 60, alpha=0.08, color="#ff9800", zorder=0)
ax.plot(T_dense, y_boot_std, "-", color="navy", lw=2, label="Bootstrap std (σ)")
ax.fill_between(T_dense, 0, y_boot_std, alpha=0.25, color="navy")
ax.axhline(SIGMA_STRESS_NOISE, color="green", ls="--", lw=1.5,
           label=f"Digitization noise floor: {SIGMA_STRESS_NOISE} MPa")
ax.axvline(100., color="red", ls=":", lw=1.5, alpha=0.8,
           label="T=100K (CRYO-CRITICAL boundary)")
ax.set_xlabel("T (K)"); ax.set_ylabel("Bootstrap std (MPa)")
ax.set_title(f"(b) Epistemic uncertainty by temperature\n"
             f"below 100K: {std_below100:.1f} MPa  |  above: {std_above100:.1f} MPa")
ax.legend(fontsize=7); ax.set_xlim(0., 510.)

ax = axes[2]
# Calibration plot: predicted CI width vs actual absolute error
# For each test point, get CI width and actual error
T_test_CIwidths = []
T_test_AbsErr   = []
for T_t, y_t, y_p in zip(T_test, y_test, y_pred_test):
    idx_nearest = np.argmin(np.abs(T_dense - T_t))
    ci_width    = y_boot_ci95_hi[idx_nearest] - y_boot_ci95_lo[idx_nearest]
    abs_err     = abs(y_t - y_p)
    T_test_CIwidths.append(ci_width)
    T_test_AbsErr.append(abs_err)

T_test_CIwidths = np.array(T_test_CIwidths)
T_test_AbsErr   = np.array(T_test_AbsErr)

# Add training points for broader calibration picture
T_all_ci = []
T_all_err= []
for T_t, y_t, y_p in zip(np.concatenate([T_train, T_test]),
                           np.concatenate([y_train, y_test]),
                           np.concatenate([y_pred_train, y_pred_test])):
    idx = np.argmin(np.abs(T_dense - T_t))
    T_all_ci.append(y_boot_ci95_hi[idx] - y_boot_ci95_lo[idx])
    T_all_err.append(abs(y_t - y_p))

ax.scatter(T_all_ci, T_all_err,
           c=[COLOR_TRAIN]*len(T_train) + [COLOR_TEST]*len(T_test),
           s=80, zorder=5, edgecolors="white", lw=0.5)
max_val = max(max(T_all_ci), max(T_all_err)) * 1.05
ax.plot([0, max_val], [0, max_val/2], "--", color="green", lw=1.5,
        label="Error = 0.5×CI width (well-calibrated)")
ax.plot([0, max_val], [0, max_val],   "-", color="grey",  lw=1, alpha=0.5,
        label="Error = CI width")
ax.set_xlabel("Bootstrap 95% CI width (MPa)")
ax.set_ylabel("|Predicted - Actual| (MPa)")
ax.set_title("(c) CI calibration plot\n(below diagonal = conservative CI)")
ax.legend(fontsize=7)
# Add point labels
for T_t, cw, ae in zip(np.concatenate([T_train, T_test]),
                        T_all_ci, T_all_err):
    ax.annotate(f"{T_t:.0f}K", (cw, ae), fontsize=6, alpha=0.7,
                xytext=(3, 2), textcoords="offset points")

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/fig_M3_uncertainty.png", bbox_inches="tight", dpi=150)
plt.close()
print(f"  [saved] fig_M3_uncertainty.png")


print("\nWriting Metrics Report")

with open(f"{OUT_DIR}/pysr_metrics_report.txt", "w") as f:
    f.write("=" * 70 + "\n")
    f.write("  PySR Symbolic Regression - Full Metrics Report\n")
    f.write("  Cryogenic SMA Empirical Stress-Curve Modeling (Song 2026 data)\n")
    f.write("=" * 70 + "\n\n")

    f.write("CONFIGURATION\n")
    f.write(f"  PySR niterations:    {PYSR_NITERATIONS}\n")
    f.write(f"  Bootstrap resamples: {N_BOOT}\n")
    f.write(f"  Digitization noise:  ±{SIGMA_STRESS_NOISE} MPa, ±{SIGMA_T_NOISE} K\n")
    f.write(f"  Physics constants:\n")
    f.write(f"    ε_tr = {EPS_TR} (Song 2025)\n")
    f.write(f"    ρ    = {RHO} kg/m³ (Song 2025)\n")
    f.write(f"    Cp   = {CP} J/kg·K (294 K consistency check only)\n\n")

    f.write("SELECTED EMPIRICAL SYMBOLIC EQUATION\n")
    f.write(f"  Form: {best_eq_str}\n")
    f.write(f"  Complexity: {best_complexity} nodes\n\n")

    f.write("TRAINING AND TEST PERFORMANCE (Ti-Al-Cr stress data, Song 2026)\n")
    f.write(f"  {'Metric':<30s}  {'PySR (M3)':>12}  {'Linear CC baseline':>18}\n")
    f.write(f"  {'-'*64}\n")
    f.write(f"  {'Train R²':<30s}  {r2_train:>12.4f}  {lr_r2_train:>18.4f}\n")
    f.write(f"  {'Test  R²':<30s}  {r2_test:>12.4f}  {lr_r2_test:>18.4f}\n")
    f.write(f"  {'Test  MAE (MPa)':<30s}  {mae_test:>12.2f}  {lr_mae_test:>18.2f}\n")
    f.write(f"  {'Test  RMSE (MPa)':<30s}  {rmse_test:>12.2f}  {lr_rmse_test:>18.2f}\n")
    f.write(f"  {'Complexity (nodes)':<30s}  {best_complexity:>12}  {'2 (2 params)':>18}\n\n")

    f.write("CC INVERSION DETECTION\n")
    f.write(f"  Reference (Song 2026): ~150 K (compression)\n")
    f.write(f"  Detected by PySR:\n")
    for T_inv in inversion_Ts:
        err = abs(T_inv - CC_INV_T)
        f.write(f"    T_inv = {T_inv:.1f} K  |  Error = {err:.1f} K  |  "
                f"{'PASS' if err < 25 else 'MARGINAL' if err < 50 else 'FAIL'}\n")
    if not inversion_Ts:
        f.write(f"    NO INVERSION DETECTED - increase niterations to {PYSR_NITERATIONS*5}\n")

    f.write("\nCROSS-ALLOY TRANSFER (coefficient refitting only)\n")
    f.write(f"  {'Alloy':<30s}  {'n':>4}  {'R²_transfer':>12}  {'MAE (MPa)':>10}\n")
    f.write(f"  {'-'*60}\n")
    for res in transfer_results:
        f.write(f"  {res['alloy']:<30s}  {res['n']:>4}  "
                f"{res['R2_transfer']:>12.4f}  {res['MAE_MPa']:>10.2f}\n")
    if not transfer_results:
        f.write("  No transfer results (excluded pending data re-digitization / LOO)\n")

    f.write("\nUNCERTAINTY QUANTIFICATION (Bootstrap N=200)\n")
    f.write(f"  95% CI coverage on test set:  {ci95_coverage:.0f}%\n")
    f.write(f"  Mean std below 100K:          {std_below100:.1f} MPa\n")
    f.write(f"  Mean std above 100K:          {std_above100:.1f} MPa\n")
    f.write(f"  {'PASS' if ci95_coverage >= 90 else 'FAIL'}: CI coverage {'>=' if ci95_coverage>=90 else '<'} 90%\n")

    f.write("\nDIRECT ΔTad MEASUREMENTS AND 294K CONSISTENCY CHECK\n")
    f.write(f"  Direct ΔTad at 294K = {dTad_measured_294:.2f} K\n")
    f.write(f"  ΔSeff/Cp check at 294K = {dTad_check_294:.2f} K\n")
    f.write("  ΔSeff/Cp comparisons are omitted below 200K.\n")
    f.write(f"  ΔSeff at 294K = {DeltaSeff_294:.2f} J/kg·K (dataset interpolation)\n")
    f.write(f"  COP estimate  = {COP_297:.2f} (Song 2025 reference: 4.6)\n")
    if T_zero_eff:
        f.write(f"  ΔSeff zero:    T ≈ {T_zero_eff[0]:.0f} K     (Song 2026: ~97 K)\n")

    f.write("\nINDEPENDENT CALORIMETRIC CLAUSIUS-CLAPEYRON TEST\n")
    f.write("  Integrate dσ_0/dT = -ΔS·ρ/ε_tr using calorimetric ΔS(T)\n")
    f.write(f"  Anchor: measured σ_0({T_anchor:.0f}K)={sigma0_anchor:.1f} MPa "
            "(excluded from error metrics)\n")
    f.write(f"  Non-anchor measured σ_0: n={len(T_cc_compare)}, "
            f"RMSE={cc_sigma_rmse:.2f} MPa, MAE={cc_sigma_mae:.2f} MPa, "
            f"R²={cc_sigma_r2:.4f}\n")

    f.write("\nBENCHMARK COMPARISON\n")
    f.write(f"  Liu 2024b (ambient NiTi, gradient boosting): R²=0.92, MAE=23.4°C\n")
    f.write(f"  Honrao 2023 (multi-alloy NiTi, 8000 pts):   MAE=14.8°C (Ts)\n")
    f.write(f"  Nespoli 2023 (Cu-Al-Mn linear regression):  linear\n")
    f.write(f"  THIS WORK - PySR (cryogenic, N={len(T_train)}):  "
            f"R²={r2_test:.4f}, MAE={mae_test:.1f}MPa\n")
    f.write("  -> Empirical symbolic approximation; not asserted as a constitutive law\n")
    f.write("\nSYMBOLIC EQUATION SELECTION\n")
    f.write("  Features: one dimensionless input, T/100K\n")
    f.write("  Pre-specified rule: maximum leave-one-out R² after refitting equation constants "
            "within each fold\n")
    f.write("  Limitation: candidate expression structures were generated using the "
            "full training set; only constants were refit in each fold, so this is "
            "not fully nested cross-validation\n")
    if pysr_expression is not None:
        f.write(f"  Selected equation LOO R²: {best_loo_r2:.4f}\n")
    f.write(f"  Fitted curve evaluated only through {T_MAX_PHYSICAL:.0f}K; "
            "no high-temperature constitutive extrapolation claimed\n")

    f.write("\nFOR PAPER TABLE 2 (comparison with M1-GPR and M2-PINN)\n")
    f.write(f"  Model M3 (PySR)\n")
    f.write(f"    R²_test           = {r2_test:.4f}\n")
    f.write(f"    MAE_test (MPa)    = {mae_test:.2f}\n")
    f.write(f"    RMSE_test (MPa)   = {rmse_test:.2f}\n")
    f.write(f"    T_inversion (K)   = "
            f"{inversion_Ts[0]:.1f}" if inversion_Ts else "NOT DETECTED")
    f.write(f"\n")
    f.write("    Cross-alloy transfer = not reported (pending source-data digitization)\n")
    f.write(f"    CI coverage (%)   = {ci95_coverage:.0f}\n")
    f.write(f"    Complexity (nodes)= {best_complexity}\n")
    f.write(f"    Interpretability  = symbolic expression (empirical approximation)\n")

print(f"  [saved] pysr_metrics_report.txt")


print("  Tension/compression comparison omitted pending digitized source data.")
# Write PySR row to unified mechanical design table
import csv as _csv
table2_path = f"{OUT_DIR}/mechanical_design_table.csv"
if os.path.exists(table2_path):
    T_inv_pysr = inversion_Ts[0] if inversion_Ts else float("nan")
    T_min_pysr = T_zero_eff[0] if T_zero_eff else float("nan")
    cop_vals_physical = COP_chain[(T_dense >= 50.) & (T_dense <= T_MAX_PHYSICAL) & np.isfinite(COP_chain)]
    T_cop_physical    = T_dense[(T_dense >= 50.) & (T_dense <= T_MAX_PHYSICAL) & np.isfinite(COP_chain)]
    T_cop_peak_pysr   = float(T_cop_physical[np.argmax(cop_vals_physical)]) if len(cop_vals_physical) else float("nan")
    COP_peak_pysr     = float(np.nanmax(cop_vals_physical)) if len(cop_vals_physical) else float("nan")
    W_pysr_vals = (sigma_pysr(T_dense) - np.interp(T_dense, df[(df["alloy_system"]=="Ti-Al-Cr")&(df["property"]=="sigma_Af")].sort_values("T_K")["T_K"].values, df[(df["alloy_system"]=="Ti-Al-Cr")&(df["property"]=="sigma_Af")].sort_values("T_K")["value"].values)) * EPS_TR * 1e6 / RHO
    W_pysr_vals = np.clip(W_pysr_vals, 0., None)
    W_physical_mask = (T_dense >= 50.) & (T_dense <= T_MAX_PHYSICAL)
    W_max_pysr  = float(np.max(W_pysr_vals[W_physical_mask]))
    T_W_max_pysr= float(T_dense[W_physical_mask][np.argmax(W_pysr_vals[W_physical_mask])])

    pysr_row = {
        "model"             : "PySR_M3",
        "T_inv_K"           : f"{T_inv_pysr:.1f}",
        "T_min_cool_K"      : f"{T_min_pysr:.1f}",
        "T_COP_peak_K"      : f"{T_cop_peak_pysr:.1f}",
        "T_max_cool_K"      : f"{T_MAX_PHYSICAL:.1f}",
        "window_width_K"    : f"{T_MAX_PHYSICAL - T_min_pysr:.1f}" if np.isfinite(T_min_pysr) else "nan",
        "W_max_Jkg"         : f"{W_max_pysr:.1f}",
        "W_max_ci_Jkg"      : "N/A (bootstrap_parametric)",
        "T_W_max_K"         : f"{T_W_max_pysr:.1f}",
        "T_H50_K"           : "N/A",
        "dTad_297K"         : "N/A (direct measurements supplied at discrete temperatures)",
        "COP_297K"          : f"{COP_297:.2f}",
        "DeltaSeff_zero_K"  : f"{T_min_pysr:.1f}",
        "T_min_ci95_K"      : f"[bootstrap_not_propagated_to_caloric]",
        "T_fatigue_onset_K" : "N/A",
    }
    with open(table2_path, newline="") as _hf:
        _fieldnames = next(_csv.reader(_hf))
    _file_exists = os.path.exists(table2_path)
    with open(table2_path, "a", newline="") as _f:
        _writer = _csv.DictWriter(_f, fieldnames=_fieldnames, extrasaction="ignore")
        _writer.writerow(pysr_row)
    print(f"\n  [appended] PySR row to {table2_path}")
else:
    print(f"\n  [WARN] {table2_path} not found - run GPR (M1) first to create table, then PySR will append.")
# 16. FINAL SUMMARY
print("\n" + "=" * 70)
print("  PySR SYMBOLIC REGRESSION - COMPLETE")
print("=" * 70)
print(f"\n  Selected empirical symbolic equation: {best_eq_str}")
print(f"  Complexity: {best_complexity} nodes")
print(f"  Test R²: {r2_test:.4f}  |  MAE: {mae_test:.2f} MPa")
print(f"  Baseline (linear CC): R²={lr_r2_test:.4f}  |  MAE={lr_mae_test:.2f} MPa")

if inversion_Ts:
    print(f"\n  CC inversion detected at T={inversion_Ts[0]:.1f} K")
    print(f"  Song 2026 reference: ~{CC_INV_T:.0f} K")
    err_pct = abs(inversion_Ts[0] - CC_INV_T) / CC_INV_T * 100
    print(f"  Error: {abs(inversion_Ts[0]-CC_INV_T):.1f} K ({err_pct:.1f}%)")
else:
    print(f"\n  WARNING: CC inversion not detected.")
    print(f"  ACTION: Increase PYSR_NITERATIONS to {PYSR_NITERATIONS*5} and retrain.")

print(f"\n  Direct ΔTad at 294K: {dTad_measured_294:.1f} K; "
      f"294K Cp consistency check: {dTad_check_294:.1f} K")
print(f"  COP estimate @ 297K: {COP_297:.2f} (Song 2025 reference: 4.6)")

print(f"\n  Bootstrap CI coverage: {ci95_coverage:.0f}% on test set")
print(f"  Independent calorimetric CC test RMSE: {cc_sigma_rmse:.2f} MPa "
      f"(n={len(T_cc_compare)}, anchor excluded)")

print(f"\n  Output files:")
for fn in [
    "pysr_equations_pareto.csv",
    "pysr_fitted_params.json",
    "pysr_metrics_report.txt",
    "fig_M3_pareto.png",
    "fig_M3_fit_and_transfer.png",
    "fig_M3_slope_analysis.png",
    "fig_M3_caloric_chain.png",
    "fig_M3_uncertainty.png",
]:
    fpath = f"{OUT_DIR}/{fn}"
    status = "ok" if os.path.exists(fpath) else "missing"
    print(f"    {status} {fpath}")

print("\n  To run with full PySR on your local machine:")
