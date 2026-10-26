# Temperature or composition? Separating them inside the inversion

Code for the ES-SSA 2026 poster (P-24) by Hesam Saeidi. It
fits a physics-informed network to the SP12RTS tomography model and
measures how much of the mantle two mineral-physics end-members (thermal
and compositional) can actually explain.

## The idea

Each grid point's P-wave anomaly is split into a thermal share `x` and a
compositional share `y`. A fixed mineral-physics layer maps them to
both wave speeds:

```
dlnVp = x + y
dlnVs = R_T·x + R_C·y        R_T = 2.00, R_C = 1.43  (Cammarano et al. 2003)
```

- **P only:** one equation and two unknowns. The loss has a flat valley,
  and every random seed lands somewhere different along it.
- **P + S:** two equations and two unknowns. The solution is unique
  whenever R_T ≠ R_C, and the seed-to-seed spread collapses.

Temperature comes from the thermal share,
`ΔT = x · R_T · Vs0(z) / (dVs/dT)`, with dVs/dT = 1.2 m/s/K (Faul &
Jackson 2005) and Vs0 from PREM. Composition is reported as an
Fe-equivalent ΔMg# = 5.714 · y, calibrated on the Cammarano et al. (2003)
pyrolite iron case.

A mix of the two end-members can only produce a point ratio
dlnVs/dlnVp between R_C and R_T. At 500 km, only about 11% of the map (by
area) falls in that band.

## Contents

| file | what it does |
|---|---|
| `pinn_degeneracy.py` | Trains the network: N seeds, P only vs P + S, at one depth. Writes a figure, an `.npz` and a row of `runs/summary.csv` |
| `sp12rts_eval.py` | Checks SP12RTS against the published Koelemeijer et al. (2016) diagnostics and draws the poster figures A–C |
| `data/` | Where the SP12RTS files go (not tracked; see `data/README.md`) |

## Setup

```bash
conda env create -f environment.yml
conda activate sp12rts-pinn
```

Then put the two SP12RTS files in `data/` (see [data/README.md](data/README.md)).

## Running

**1. Train the network.** The poster run is 500 km, every 6th grid point,
50 seeds and 4000 Adam steps:

```bash
python pinn_degeneracy.py --depth 500 --sub 6 --seeds 50 --steps 4000 --tag w_base
```

Every setting is a command-line flag (`python pinn_degeneracy.py -h`).
Outputs go to `runs/`, named after the settings. Each run adds a row to
`runs/summary.csv`, so runs can be compared directly.

`--init-scale` multiplies the initial weights. It is a diagnostic: the
default of 1.0 reproduces the runs above, and the poster uses ×3 and ×10
as sensitivity checks.

```bash
python pinn_degeneracy.py --depth 500 --sub 6 --seeds 50 --steps 4000 --init-scale 3  --tag init3
python pinn_degeneracy.py --depth 500 --sub 6 --seeds 50 --steps 4000 --init-scale 10 --tag init10
```

**2. Evaluate SP12RTS and draw the figures.** Figures B and C read the
`w_base` run from step 1.

```bash
python sp12rts_eval.py data/SP12RTS_dvs.nc data/SP12RTS_dvp.nc --out eval
```

| output | figure |
|---|---|
| `eval.png` | depth profiles: RMS amplitude, R = dlnVs/dlnVp, correlations |
| `eval_mapsA_d500.png`, `_d700` | **A.** dlnVp over dlnVs, Robinson projection, one shared ±1.2% colour scale |
| `eval_seedsB.png` | **B.** recovered ΔT per seed at CVL, and the spread across seeds at every grid point |
| `eval_fieldsC.png` | **C.** recovered ΔT and ΔMg# (P + S), greyed where the point ratio is outside [R_C, R_T] |

## Checks

Each figure prints what it should show, so a wrong figure can be caught
without knowing in advance what it should look like:

- **Profile figure:** 7/7 Koelemeijer et al. (2016) checks pass, including
  R peaking at 2300–2700 km and decreasing through D″.
- **A:** the dlnVp panel is visibly paler than the dlnVs panel. If both look
  equally colourful, the colour scale is not shared.
- **B:** P + S gives ΔT ≈ +108.5 ± 3.1 K at CVL. The median spread is about
  31.9 K for P only and 3.9 K for P + S.
- **C:** about 11% of the area is coloured. More than about 20% means the
  mask is inverted. CVL falls in the grey.
- **PINN:** observed R at 500 km is 2.2929 (area-weighted, every 6th point).

## Notes

- **Area weighting:** the loss, the misfit and R are all weighted by
  cos(latitude), so the poles don't dominate.
- **Composition comes from the network:** `y` is the network's own output.
  It is never recomputed as `dlnVp − x`, which would make corr(x, y) = −1
  by construction.
- **Float32 data:** SP12RTS is stored as float32, so identities such as
  `x + y = dlnVp` hold only to float32 precision.
- **The readout point:** "CVL" is 3°N, 10°E snapped onto the every-6th-point
  grid, which gives 0°N, 12°E.

## References

- Cammarano, F., Goes, S., Vacher, P. & Giardini, D. (2003). Inferring
  upper-mantle temperatures from seismic velocities. *PEPI* 138, 197–222.
- Faul, U. H. & Jackson, I. (2005). The seismological signature of
  temperature and grain size variations in the upper mantle. *EPSL* 234,
  119–134.
- Koelemeijer, P., Ritsema, J., Deuss, A. & van Heijst, H.-J. (2016).
  SP12RTS: a degree-12 model of shear- and compressional-wave velocity for
  Earth's mantle. *GJI* 204, 1024–1039.
- Tancik, M. et al. (2020). Fourier features let networks learn high
  frequency functions in low dimensional domains. *NeurIPS*.

## Contact

Hesam Saeidi · saeidi.hesam@gmail.com
