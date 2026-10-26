#!/usr/bin/env python3
"""
pinn_degeneracy.py — physics-informed network, E1 (P only) vs E2 (P + S).

Every setting is a command-line flag. Nothing needs editing between runs.
Each run writes files tagged with its settings and appends a row to
runs/summary.csv so runs can be compared directly.

EXAMPLES
    python pinn_degeneracy.py
    python pinn_degeneracy.py --seeds 50 --sub 1 --steps 4000
    python pinn_degeneracy.py --depth 535 --rt 2.4 --rc 1.5 --tag mtzcentre
    python pinn_degeneracy.py --vs path/to/dvs.nc --vp path/to/dvp.nc

ARCHITECTURE
    (Fourier features of lat, lon)
        -> MLP (tanh)                         <- the only part that learns
    -> (x, y) = thermal and compositional shares of the P-wave anomaly
        -> fixed mineral-physics layer        <- algebra, never learns
    -> dlnVp = x + y,   dlnVs = R_T*x + R_C*y
    -> compared with SP12RTS

    Temperature from the thermal share:
        dT = x * R_T * Vs0(z) / (dVs/dT),  dVs/dT = 1.2 m/s/K
        (Faul & Jackson 2005, as used in Text S4 of Saeidi et al. 2023)
        Vs0(z) is PREM Vs interpolated at the model depth.

    --rt and --rc default to R_T = 2.00 and R_C = 1.43 (Cammarano et al.
    2003). R_T > R_C because anelasticity makes Vs more temperature-
    sensitive than Vp.

THE POINT
    Fit dlnVp alone -> one equation, two unknowns. The loss has a flat
    valley; every seed lands somewhere different along it. The spread is
    the result.
    Fit both        -> two equations, two unknowns, unique whenever
    R_T != R_C. The spread collapses.
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------- published constants
DVS_DT = 1.2        # m/s/K  Faul & Jackson (2005)

# PREM Vs, m/s. Table from sp12rts_eval.py, plus both sides of the
# 400 and 670 discontinuities so np.interp doesn't smear them.
PREM = np.array([[ 220, 4.64], [ 400, 4.77], [ 400.001, 4.93], [ 600, 5.52],
                 [ 670, 5.57], [ 670.001, 5.95], [ 771, 6.24], [1000, 6.38],
                 [1500, 6.72], [2000, 6.99], [2500, 7.18], [2891, 7.26]])

def vs0(z):
    return float(np.interp(z, PREM[:, 0], PREM[:, 1]) * 1000.0)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e2e0da"

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": INK2, "axes.linewidth": 0.9, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


# --------------------------------------------------------------------- CLI
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Degeneracy experiment on SP12RTS (E1 vs E2).")
    p.add_argument("--vs", help="path to SP12RTS_dvs.nc (auto-found if absent)")
    p.add_argument("--vp", help="path to SP12RTS_dvp.nc (auto-found if absent)")
    p.add_argument("--depth", type=float, default=500.0,
                   help="target depth in km (snaps to nearest model level)")
    p.add_argument("--sub", type=int, default=6,
                   help="use every Nth lat/lon point; 1 = full resolution")
    p.add_argument("--seeds", type=int, default=20,
                   help="number of random seeds per experiment")
    p.add_argument("--steps", type=int, default=1500, help="Adam steps")
    p.add_argument("--hidden", type=int, default=32, help="neurons per layer")
    p.add_argument("--kmax", type=int, default=6,
                   help="Fourier feature order (Tancik et al. 2020)")
    p.add_argument("--lr", type=float, default=6e-3, help="learning rate")
    p.add_argument("--rt", type=float, default=2.00,
                   help="R_T, Vs/Vp ratio for a purely THERMAL anomaly "
                        "(Cammarano et al. 2003)")
    p.add_argument("--rc", type=float, default=1.43,
                   help="R_C, Vs/Vp ratio for a purely COMPOSITIONAL anomaly "
                        "(Cammarano et al. 2003)")
    p.add_argument("--init-scale", type=float, default=1.0,
                   help="multiplier on the initial network weights. "
                        "Diagnostic only; 1.0 reproduces previous runs.")
    p.add_argument("--readout-lat", type=float, default=3.0,
                   help="latitude of the single point shown in panel a")
    p.add_argument("--readout-lon", type=float, default=10.0,
                   help="longitude of that point")
    p.add_argument("--readout-name", default="CVL",
                   help="label for that point")
    p.add_argument("--outdir", default=None,
                   help="output folder (default: ./runs next to this script)")
    p.add_argument("--tag", default="", help="extra label for the filenames")
    p.add_argument("--quiet", action="store_true", help="no progress lines")
    return p.parse_args()


def find_data(a):
    if a.vs and a.vp:
        return a.vs, a.vp
    here = Path(__file__).resolve().parent
    for base in [here / "data", here]:
        hits = list(base.rglob("SP12RTS_dvs.nc"))
        if hits:
            d = hits[0].parent
            return str(d / "SP12RTS_dvs.nc"), str(d / "SP12RTS_dvp.nc")
    sys.exit("Could not find SP12RTS_dvs.nc. Put it in data/ "
             "(see data/README.md) or pass --vs and --vp.")


def run_tag(a, depth_actual):
    bits = [f"d{depth_actual:.0f}", f"sub{a.sub}", f"n{a.seeds}",
            f"st{a.steps}", f"h{a.hidden}", f"k{a.kmax}",
            f"rt{a.rt:g}", f"rc{a.rc:g}"]
    if a.init_scale != 1.0:
        bits.append(f"is{a.init_scale:g}")
    if a.tag:
        bits.append(a.tag)
    return "pinn_" + "_".join(bits)


# ----------------------------------------------------------- the network
def init(rng, nin, nh, nout):
    def layer(i, o):
        return (rng.standard_normal((i, o)) * np.sqrt(1.0 / i), np.zeros(o))
    W1, b1 = layer(nin, nh)
    W2, b2 = layer(nh, nh)
    W3, b3 = layer(nh, nout)
    return [W1, b1, W2, b2, W3, b3]


def forward(params, X):
    W1, b1, W2, b2, W3, b3 = params
    a1 = np.tanh(X @ W1 + b1)
    a2 = np.tanh(a1 @ W2 + b2)
    return a2 @ W3 + b3, (a1, a2)


def backward(params, X, cache, dout):
    W1, b1, W2, b2, W3, b3 = params
    a1, a2 = cache
    gW3, gb3 = a2.T @ dout, dout.sum(0)
    dz2 = (dout @ W3.T) * (1 - a2 ** 2)
    gW2, gb2 = a1.T @ dz2, dz2.sum(0)
    dz1 = (dz2 @ W2.T) * (1 - a1 ** 2)
    gW1, gb1 = X.T @ dz1, dz1.sum(0)
    return [gW1, gb1, gW2, gb2, gW3, gb3]


def train(X, p_obs, s_obs, Wn, seed, use_s, a, tag="", verbose=True):
    rng = np.random.default_rng(seed)
    params = [q * a.init_scale for q in init(rng, X.shape[1], a.hidden, 2)]
    m = [np.zeros_like(q) for q in params]
    v = [np.zeros_like(q) for q in params]
    b1_, b2_, eps = 0.9, 0.999, 1e-8
    N = X.shape[0]

    for t in range(1, a.steps + 1):
        out, cache = forward(params, X)
        x, y = out[:, 0], out[:, 1]

        rp = x + y - p_obs
        dp = 2.0 * rp * Wn / N
        if use_s:
            rs = a.rt * x + a.rc * y - s_obs
            ds = 2.0 * rs * Wn / N
        else:
            rs, ds = 0.0, 0.0

        dout = np.empty_like(out)
        dout[:, 0] = dp + (ds * a.rt if use_s else 0.0)
        dout[:, 1] = dp + (ds * a.rc if use_s else 0.0)

        if verbose and (t == 1 or t % 250 == 0 or t == a.steps):
            loss = float((rp ** 2 * Wn).mean())
            if use_s:
                loss += float((rs ** 2 * Wn).mean())
            print(f"\r    {tag}  [{100.0*t/a.steps:5.1f}%]  "
                  f"step {t:5d}/{a.steps}  loss {loss:.3e}",
                  end="", flush=True)

        grads = backward(params, X, cache, dout)
        for i, (q, g) in enumerate(zip(params, grads)):
            m[i] = b1_ * m[i] + (1 - b1_) * g
            v[i] = b2_ * v[i] + (1 - b2_) * g * g
            mh = m[i] / (1 - b1_ ** t)
            vh = v[i] / (1 - b2_ ** t)
            params[i] = q - a.lr * mh / (np.sqrt(vh) + eps)

    if verbose:
        print()
    out, _ = forward(params, X)
    x, y = out[:, 0], out[:, 1]
    return x, y, float(np.sqrt((((x + y - p_obs) ** 2) * Wn).mean()))


# ------------------------------------------------------------------- main
def main():
    a = parse_args()
    vs_path, vp_path = find_data(a)
    print(f"data:\n  {vs_path}\n  {vp_path}")

    ds_s, ds_p = xr.open_dataset(vs_path), xr.open_dataset(vp_path)
    depth_all = ds_s["depth"].values
    i = int(np.argmin(np.abs(depth_all - a.depth)))
    depth = float(depth_all[i])

    lat = ds_s["latitude"].values[::a.sub]
    lon = ds_s["longitude"].values[::a.sub]
    s2d = ds_s["v"].isel(depth=i).values[::a.sub, ::a.sub]
    p2d = ds_p["v"].isel(depth=i).values[::a.sub, ::a.sub]

    LA, LO = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    feats = []
    for k in range(1, a.kmax + 1):
        feats += [np.sin(k * LA).ravel(), np.cos(k * LA).ravel(),
                  np.cos(k * LO).ravel(), np.sin(k * LO).ravel()]
    X = np.stack(feats, axis=1)
    p_obs, s_obs = p2d.ravel(), s2d.ravel()

    # area weights: grid cells scale as cos(latitude)
    Wn = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], s2d.shape).ravel()
    Wn = Wn / Wn.mean()          # mean 1, so N stays the natural normaliser
    assert abs(Wn.mean() - 1.0) < 1e-12, Wn.mean()

    R_obs = float(np.sqrt((Wn * s_obs ** 2).mean() / (Wn * p_obs ** 2).mean()))
    tag = run_tag(a, depth)
    outdir = Path(a.outdir) if a.outdir else Path(__file__).resolve().parent / "runs"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\nrun  {tag}")
    print(f"  depth {depth:.0f} km (asked {a.depth:g})   {p_obs.size} points "
          f"(sub={a.sub})")
    print(f"  seeds {a.seeds}   steps {a.steps}   hidden {a.hidden}   "
          f"kmax {a.kmax}   lr {a.lr:g}")
    print(f"  R_T={a.rt}  R_C={a.rc}   observed R = {R_obs:.3f}   "
          f"Vs0 (PREM) = {vs0(depth):.0f} m/s\n")

    if abs(a.rt - a.rc) < 1e-9:
        sys.exit("R_T equals R_C: the two mechanisms are indistinguishable "
                 "and no amount of S data can separate them.")

    x_true = (s_obs - a.rc * p_obs) / (a.rt - a.rc)

    def to_T(x):
        return x / 100.0 * a.rt * vs0(depth) / DVS_DT

    def to_Mg(y):
        """Change in Mg# from the compositional share.
        Calibration: Cammarano et al. (2003) Table 2 gives pyrolite
        X_Fe = 0.11 (Mg# 89); their Fe case is Mg# 89 -> 93, i.e.
        dX_Fe = -0.04, which produces dlnVp = +0.7%. Hence
        dMg# = -100 * (-0.04/0.007) * (y/100) = 5.714 * y  with y in percent."""
        return 5.714 * y

    res, res_y, misfits = {}, {}, {}
    for label, use_s in [("E1  P only", False), ("E2  P + S", True)]:
        Xs, Ys, mis = [], [], []
        for sd in range(a.seeds):
            x, y, mp = train(X, p_obs, s_obs, Wn, sd, use_s, a,
                             tag=f"{label}  seed {sd+1}/{a.seeds}",
                             verbose=not a.quiet)
            Xs.append(x)
            Ys.append(y)
            mis.append(mp)
        res[label] = np.array(Xs)
        res_y[label] = np.array(Ys)
        # diagnostic: x_net + y_net - p_obs is the Vp residual, not zero;
        # its weighted RMS must equal the misfit train() reported per seed
        rms = np.sqrt((((res[label] + res_y[label] - p_obs) ** 2) * Wn).mean(axis=1))
        assert np.allclose(rms, mis, rtol=1e-6, atol=0.0), (rms, mis)
        misfits[label] = (float(np.mean(mis)), float(np.std(mis)))
        T = to_T(res[label])
        print(f"{label}:  Vp RMS misfit = {misfits[label][0]:.4f}% "
              f"(sd {misfits[label][1]:.5f})")
        print(f"            seed-to-seed sd of dT, median over grid = "
              f"{np.median(T.std(0)):.1f} K")

    ilat = int(np.argmin(np.abs(lat - a.readout_lat)))
    ilon = int(np.argmin(np.abs(lon - a.readout_lon)))
    k = ilat * lon.size + ilon
    print(f"\nAt {a.readout_name} ({lat[ilat]:.0f}, {lon[ilon]:.0f}): "
          f"dlnVp = {p_obs[k]:+.3f}%, dlnVs = {s_obs[k]:+.3f}%")
    print(f"  analytic joint solution: dT = {to_T(x_true[k]):+.1f} K")
    stats = {}
    for label in res:
        T = to_T(res[label][:, k])
        stats[label] = (float(T.mean()), float(T.std()))
        print(f"  {label}: dT = {T.mean():+7.1f} +/- {T.std():.1f} K   "
              f"(range {T.min():+.0f} to {T.max():+.0f})")

    y_E1 = res_y["E1  P only"]
    y_E2 = res_y["E2  P + S"]
    y_true = p_obs - x_true      # algebra, not a network output
    # SP12RTS is float32: allow a few ulp of the largest |dlnVp|
    assert np.allclose(y_true + x_true, p_obs, rtol=0.0,
                       atol=4 * np.finfo(p_obs.dtype).eps * np.abs(p_obs).max())

    sd1 = float(np.median(to_T(res["E1  P only"]).std(0)))
    sd2 = float(np.median(to_T(res["E2  P + S"]).std(0)))
    print(f"\n  COLLAPSE FACTOR: {sd1/sd2:.1f}x   ({sd1:.1f} K -> {sd2:.1f} K)")

    # --------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 6.4))
    rng = np.random.default_rng(1)
    for j, (label, col) in enumerate([("E1  P only", ORANGE),
                                      ("E2  P + S", BLUE)]):
        T = to_T(res[label][:, k])
        jit = (rng.random(T.size) - 0.5) * 0.26
        ax[0].scatter(np.full(T.size, j) + jit, T, s=80, color=col,
                      edgecolor="white", lw=1.4, zorder=4)
    ax[0].axhline(to_T(x_true[k]), color=INK, lw=1.8, ls=(0, (5, 3)), zorder=3)
    ax[0].text(-0.46, to_T(x_true[k]) + 2.0, "exact solution of the joint system",
               fontsize=10, color=INK, va="bottom", ha="left", fontweight="bold")
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels([f"trained on\n$V_P$ only", f"trained on\n$V_P$ + $V_S$"])
    ax[0].set_xlim(-0.5, 1.5)
    ax[0].set_ylabel(f"recovered ΔT at {a.readout_name} (K)", fontweight="bold")
    ax[0].set_title(f"a  {a.seeds} seeds, same data", loc="left",
                    fontweight="bold", fontsize=12.5, pad=12)

    for label, col in [("E1  P only", ORANGE), ("E2  P + S", BLUE)]:
        ax[1].hist(to_T(res[label]).std(0), bins=60, color=col, alpha=0.65,
                   label=label, zorder=3)
    ax[1].set_xlabel("seed-to-seed spread in ΔT (K), per grid point",
                     fontweight="bold")
    ax[1].set_ylabel("number of grid points", fontweight="bold")
    ax[1].set_yscale("log")
    ax[1].legend(frameon=False, fontsize=10.5)
    ax[1].set_title(f"b  whole map — collapse {sd1/sd2:.1f}×", loc="left",
                    fontweight="bold", fontsize=12.5, pad=12)

    for b in ax:
        b.grid(color=GRID, lw=0.7, zorder=0)
        b.set_axisbelow(True)
        for sp in ("top", "right"):
            b.spines[sp].set_visible(False)

    fig.suptitle("Fitting P alone leaves temperature undetermined; "
                 "adding S pins it down",
                 x=0.012, ha="left", fontsize=13.5, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0.075, 1, 0.95))
    fig.text(0.012, 0.042,
             f"SP12RTS {depth:.0f} km  |  sub={a.sub} ({p_obs.size} pts)  |  "
             f"seeds={a.seeds}  steps={a.steps}  hidden={a.hidden}  "
             f"kmax={a.kmax}  lr={a.lr:g}  |  "
             f"R_T={a.rt:g}  R_C={a.rc:g}  |  observed R={R_obs:.2f}",
             fontsize=8.8, color=INK2, family="monospace")
    fig.text(0.012, 0.012,
             f"ΔT from dVs/dT = 1.2 m/s/K (Faul & Jackson 2005), "
             f"Vs0 = {vs0(depth):.0f} m/s (PREM).  "
             f"R_T, R_C from Cammarano et al. (2003).  "
             f"{tag}  {datetime.now():%Y-%m-%d %H:%M}",
             fontsize=8.2, color=MUTED)

    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"{tag}.{ext}", dpi=200, bbox_inches="tight",
                    pad_inches=0.25)
    np.savez_compressed(outdir / f"{tag}.npz",
                        lat=lat, lon=lon, depth=depth,
                        x_E1=res["E1  P only"], x_E2=res["E2  P + S"],
                        x_true=x_true, p_obs=p_obs, s_obs=s_obs,
                        y_E1=y_E1, y_E2=y_E2, y_true=y_true,
                        vs0=vs0(depth), refmodel="prem",
                        readout_index=k, **{f"arg_{kk}": vv
                                            for kk, vv in vars(a).items()
                                            if vv is not None})

    # ----------------------------------------------------- summary.csv row
    csv_path = outdir / "summary.csv"
    row = {
        "timestamp": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "tag": tag,
        "depth_km": round(depth, 1), "sub": a.sub, "n_points": p_obs.size,
        "seeds": a.seeds, "steps": a.steps, "hidden": a.hidden,
        "kmax": a.kmax, "lr": a.lr, "R_T": a.rt, "R_C": a.rc,
        "vs0_mps": round(vs0(depth), 1), "refmodel": "prem",
        "R_observed": round(R_obs, 4),
        "misfit_E1_pct": round(misfits["E1  P only"][0], 5),
        "misfit_E2_pct": round(misfits["E2  P + S"][0], 5),
        # share units (% of dlnVp): independent of Vs0 and DVS_DT
        "sd_x_E1": round(float(np.median(res["E1  P only"].std(0))), 4),
        "sd_x_E2": round(float(np.median(res["E2  P + S"].std(0))), 4),
        "sd_dT_E1_K": round(sd1, 2), "sd_dT_E2_K": round(sd2, 2),
        "collapse_factor": round(sd1 / sd2, 2),
        "readout": a.readout_name,
        "dT_exact_K": round(float(to_T(x_true[k])), 2),
        "dT_E1_K": round(stats["E1  P only"][0], 2),
        "dT_E1_sd_K": round(stats["E1  P only"][1], 2),
        "dT_E2_K": round(stats["E2  P + S"][0], 2),
        "dT_E2_sd_K": round(stats["E2  P + S"][1], 2),
        "dMg_exact": round(float(to_Mg(y_true[k])), 3),
        "dMg_E2": round(float(to_Mg(y_E2[:, k]).mean()), 3),
        "dMg_E2_sd": round(float(to_Mg(y_E2[:, k]).std()), 3),
        "weighted": True,
    }
    if a.init_scale != 1.0:
        row["init_scale"] = a.init_scale
    # If an older summary.csv has different columns, rewrite it with the
    # union of both headers (missing cells left blank) so appended rows
    # never land under the wrong column.
    old_rows, fields = [], list(row)
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            old_fields = rd.fieldnames or []
            old_rows = list(rd)
        fields = old_fields + [f for f in row if f not in old_fields]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        wr.writerows(old_rows)
        wr.writerow(row)

    print(f"\n  wrote {outdir / tag}.pdf / .png / .npz")
    print(f"  appended a row to {csv_path}")


if __name__ == "__main__":
    main()
