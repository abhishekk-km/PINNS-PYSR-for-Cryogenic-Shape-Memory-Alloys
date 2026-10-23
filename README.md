# Physics-informed surrogates and symbolic regression for the Clausius-Clapeyron inversion in cryogenic Ti-Al-Cr shape memory alloys

Code and digitised data accompanying the manuscript "Physics-Informed Surrogates and Symbolic Regression for the
Anomalous Clausius-Clapeyron Inversion in Cryogenic Ti-Al-Cr Shape Memory Alloys" (Mishra and Gautam, Department of
Mechanical Engineering, Delhi Technological University).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

Archived release (code, digitised data and model outputs): https://doi.org/10.5281/zenodo.XXXXXXX
Development repository: https://github.com/abhishekk-km/PINNS-PYSR-for-Cryogenic-Shape-Memory-Alloys

## What this repository does

Single-crystal Ti-Al-Cr keeps superelasticity down to 4.2 K because its transformation stress passes through a minimum
and the Clausius-Clapeyron (CC) slope changes sign. This code reproduces that behaviour with five complementary models
and carries it through the dissipation chain to cooling performance.

| Model | Role | Script |
|---|---|---|
| Landau-Devonshire potential with a Maxwell construction | Mechanism: two competing temperature channels give the slope reversal | `02_landau_fit.py` |
| Gaussian process (GP), Barrett-GP, Landau-informed GP | Calibrated uncertainty, leave-one-out and withheld-window tests | `02_landau_fit.py`, `04_gp_and_validation.py` |
| Symbolic regression (PySR) | Closed-form expression for the upper-plateau stress | `01_symbolic_regression.py` |
| Hard-constraint physics-informed network | Stress, entropy, dissipation and adiabatic temperature change tied by construction | `03_hard_constraint_network.py` |
| Debye heat capacity and kappa calibration | Temperature-dependent Cp and dissipation amplification | `02_landau_fit.py` |

### Governing relations used

- Clausius-Clapeyron: `d(sigma_0)/dT = -(rho / eps_tr) * dS * 1e-6` (MPa/K, dS in J/(kg K))
- Effective entropy: `dS_eff = dS + kappa * Q_diss / (2T)`
- Adiabatic temperature change: `dT_ad = T * dS_eff / Cp(T)`
- Material coefficient of performance: `COP_mat = T * |dS_eff| / Q_diss` (valid where `dS_eff < 0`)
- Landau free energy: `f = a/2 eta^2 - b/4 eta^4 + c/6 eta^6 - sigma * eps_tr * eta`, reduced to one universal Maxwell
  curve `s(alpha)` with fixed points `alpha_M = 3/16` and `alpha_C = 9/20`

Constants for Ti-Al-Cr: transformation strain `eps_tr = 0.064`, density `rho = 4500 kg/m^3`, room-temperature specific
heat `Cp = 588 J/(kg K)`.

## Main results reported in the manuscript

- The Landau model places the stress minimum at 135 K with limiting slopes of -2.2 and +1.4 MPa/K.
- Held-out stress error: GP 0.9 percent, Landau-informed GP 3.6 percent, symbolic regression 4.3 percent. The three
  place the slope reversal at 150, 135 and 164 K.
- The hard-constraint network predicts held-out stress, entropy and temperature change to 3 to 4 percent, the cooling
  onset at 96.1 K and a material COP of 4.4.
- A Debye heat capacity roughly halves the temperature-change error below room temperature.
- Comparing the Landau entropy with calorimetry shows an excess of 20 to 56 J/(kg K) below 150 K.

Exact values are written to `outputs/landau/paper_metrics.json`, `outputs/pinn_hard/pinn_hard_metrics.json` and
`outputs/final/results.json` when the scripts are run.

## Repository layout

```
.
|-- 01_symbolic_regression.py        PySR search, leave-one-out equation selection, bootstrap bands
|-- 02_landau_fit.py                 Maxwell construction, Landau fit, GP variants, kappa, Debye Cp, Monte Carlo
|-- 03_hard_constraint_network.py    Ensemble network with the dissipation chain built into the architecture
|-- 04_gp_and_validation.py          GP posterior, withheld-window tests, CC violation test, interval calibration
|-- 05_make_figures.py               Manuscript figures 1, 2 and 4
|-- 06_overview_figure.py            Manuscript overview figure
|-- digitize_figure.py               Mouse-driven digitiser used to read curves from a figure image
|-- data/dataset.csv                 Digitised experimental values with source and uncertainty
|-- outputs/                         Metrics, parameters and predictions behind the reported numbers
|-- figures/                         Figures used in the manuscript
|-- requirements.txt
`-- LICENSE
```

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

PySR needs Julia. PySR installs it on first use, which can take several minutes. Package versions in
`requirements.txt` are those used to produce the reported results.

## Running the analysis

Run from the repository root, in this order. Later steps read files written by earlier ones.

```bash
python 01_symbolic_regression.py        # writes outputs/pysr_equations_pareto.csv and the PySR report
python 02_landau_fit.py                 # writes outputs/landau/ (parameters, Maxwell table, paper_metrics.json)
python 03_hard_constraint_network.py    # needs outputs/landau/, writes outputs/pinn_hard/
python 04_gp_and_validation.py          # writes outputs/final/
python 05_make_figures.py               # needs the three output folders above
python 06_overview_figure.py            # needs outputs/landau/
```

Quick smoke test of steps 2 to 4 with reduced epochs and iterations:

```bash
FAST=1 python 02_landau_fit.py
FAST=1 python 03_hard_constraint_network.py
FAST=1 python 04_gp_and_validation.py
```

To use a different dataset file, set `SMA_DATA=path/to/file.csv`. To rerun PySR from scratch, delete
`outputs/pysr_tiAlCr_model.pkl`. Random seeds are fixed in every script (42 for most, documented where different), but
PySR, PyTorch and library versions can still change the last digits.

## Data

`data/dataset.csv` contains values digitised from published figures and quoted from published text. No new
experimental data were generated in this work.

| Column | Meaning |
|---|---|
| `paper_id`, `author_year` | Source publication |
| `alloy_system`, `composition`, `processing` | Material and condition |
| `T_K` | Temperature in kelvin |
| `property` | Quantity name, for example `sigma_Ms`, `sigma_Af`, `sigma_0`, `Qdiss`, `DeltaSeff`, `DeltaS_cal`, `dTad_meas` |
| `value`, `unit` | Value and unit |
| `split` | `train`, `test` or `validation` (fixed before any model was fitted) |
| `cryo_flag` | `CRYO-CRITICAL` below 150 K, `EXTREME-CRYO` below 50 K |
| `notes` | Source figure and the stated reading uncertainty |

Sources of the digitised values and quoted scalars:

| Alloy family | Source | DOI |
|---|---|---|
| Ti-Al-Cr | Song et al., Nat. Commun. 17 (2026) | 10.1038/s41467-026-72172-7 |
| Ti-Al-Cr | Song et al., Nature 638, 965-971 (2025) | 10.1038/s41586-024-08583-7 |
| Ti-Ni-Fe | Zeng et al., Scripta Mater. 272 (2026) | 10.1016/j.scriptamat.2025.117075 |
| Mg-Sc | Sun et al., Mater. Sci. Eng. A 958 (2026) | 10.1016/j.msea.2026.149991 |
| Cu-Al-Mn | Nespoli et al., Metals (2023) | 10.3390/met13071311 |

The `paper_id` and `author_year` columns give the source of every row. Ti-Al-Cr is the primary alloy. The other
families provide context and feed the multi-alloy sections of `04_gp_and_validation.py`.

Digitisation uncertainty is taken as +/-2 MPa in stress and +/-3 K in temperature. Because most curves are read from the
same source paper as the reference values, comparisons against that paper test internal consistency and not
independent prediction. The original figures and measurements belong to the cited publications. Please cite the
original sources when reusing any row.

## Known limitations

- All Ti-Al-Cr data are single-crystal compression measurements. Polycrystalline or tensile behaviour needs
  recalibration.
- Stress data end at 25 K and caloric data at about 48 K. Statements below those temperatures are extrapolations.
- The dissipation amplification factor kappa and the CC conversion factor are effective parameters fitted to the same
  source curves. They absorb uncertainty in rho, eps_tr and the loop area.
- The quantum-saturation temperature of the order-parameter channel is only weakly constrained by the fit. The
  conclusions rest on the structure of the two-channel driving function and not on that number.
- The symbolic-regression structure search uses all training data. Only the constants are refitted inside each
  cross-validation fold, so the selection is not fully nested.

## Citation

If you use this code or data, please cite the archived release:

Mishra, A. K. and Gautam, V. Code and digitised data for physics-informed surrogates of the Clausius-Clapeyron
inversion in cryogenic Ti-Al-Cr shape memory alloys. 

GitHub also offers a "Cite this repository" button, which reads `CITATION.cff`. Please also cite the manuscript
(reference to be added on acceptance) and the original experimental sources listed above.

## Licence

Code: MIT licence (see `LICENSE`). Dataset compilation (`data/dataset.csv`): CC BY 4.0 for the selection, structure and
annotations, with the values themselves attributed to the cited publications. Reuse of any row requires citing the
original source.

## Contact

Abhishek Kumar Mishra, Department of Mechanical Engineering, Delhi Technological University.
Corresponding author: Vijay Gautam (vijaygautam@dce.ac.in).