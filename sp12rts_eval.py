#!/usr/bin/env python3
"""
sp12rts_eval.py — dlnVs / dlnVp evaluation of SP12RTS.

Reproduces the published diagnostics of Koelemeijer, Ritsema, Deuss & van
Heijst (2016), GJI 204, 1024, as a verification of the model files AND as the
starting science figure for the poster.

Computes, per depth layer:
  - area-weighted RMS of dlnVs and dlnVp
  - R = dlnVs/dlnVp by three estimators:
        (a) RMS ratio                  [the paper's preferred cross-model one]
        (b) median of point ratios, masked at |dlnV| < 0.1%   [their recipe]
        (c) regression bracket [R_b1, R_b2]:
              R_b1 = <sp>/<p^2>  regress S on P (Vp exact)  -> lower bound
              R_b2 = <s^2>/<sp>  regress P on S (Vs exact)  -> upper bound
            R_b2/R_b1 = 1/r0^2 (r0 = uncentred correlation) and
            R_rms = sqrt(R_b1 * R_b2) exactly, so the band brackets R_rms.
  - corr(dlnVs, dlnVp)
  - dlnVc (bulk sound) and corr(dlnVs, dlnVc)

Expected fingerprint:
  R rises through the lower mantle, peaks near 2500 km (2.7 median, up to 5.4
  by regression), then DECREASES through the lowermost ~300 km. The authors
  call that decrease "a robust feature of SP12RTS" and note it is not a clear
  characteristic of other models - so it is the discriminating test.
  Also: dlnVs-dlnVc correlation turns negative in the lower mantle below
  ~1500 km.

Poster figures (written after the profile figure):
  A  <out>_mapsA_d<z>   dlnVp | dlnVs maps per --depths, Robinson, coastlines,
                        ONE shared colour scale (+/- --vmax). Needs cartopy.
     VERIFY: left (dlnVp) panel visibly paler than right (dlnVs). If they look
     equally colourful the scale is not shared and R has been normalised away.
  B  <out>_seedsB       per-seed figure from the weighted PINN .npz (--npz).
     VERIFY: P + S mean ~ +108.5 +/- 3.1 K at CVL, P only spread ~ 30 K,
     median spreads 31.9 K and 3.9 K.
  C  <out>_fieldsC      recovered dT and dMg# maps from the same .npz, grey
                        where the point ratio dlnVs/dlnVp is outside [R_C, R_T].
                        Needs cartopy.
     VERIFY: printed area-weighted coloured fraction ~ 11%. More than ~20%
     means the mask is inverted. CVL must land in the grey.

Usage:  python sp12rts_eval.py data/SP12RTS_dvs.nc data/SP12RTS_dvp.nc [--out prefix] [--mask-above KM]
                               [--depths 500 700] [--npz run.npz]

  --mask-above KM  sets all R estimators to NaN for depths < KM. Use it for
  the shallowest spline layers, where Koelemeijer et al. (2016) document
  Vp <-> Vs projection (R ~ 1 by construction, not by physics). Pick KM from
  the printed table: the depth where R(rms) stops sitting at ~1.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e2e0da"
GREY = "#d4d2cc"                                   # masked points, figure C

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.9, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# PREM Vp, Vs (km/s) at selected depths (km), for the bulk-sound conversion.
# Linear interpolation between these is adequate for a correlation diagnostic.
PREM_DEPTH = np.array([   0,  24,  80, 220, 400, 600, 670, 771,
                       1000, 1500, 2000, 2500, 2741, 2891])
PREM_VP    = np.array([5.80, 6.80, 8.08, 8.56, 9.13,10.16,10.75,11.07,
                      11.46,12.18,12.78,13.27,13.62,13.72])
PREM_VS    = np.array([3.20, 3.90, 4.47, 4.64, 4.93, 5.52, 5.95, 6.24,
                       6.38, 6.72, 6.99, 7.18, 7.26, 7.26])

DVS_DT = 1.2        # m/s/K  Faul & Jackson (2005), as in pinn_degeneracy.py
MG_PER_Y = 5.714    # dMg# per % of y, Cammarano et al. (2003); see to_Mg()


def area_weights(lat):
    w = np.cos(np.deg2rad(lat))
    return w / w.sum()


def style(b):
    b.grid(color=GRID, lw=0.7, zorder=0)
    b.set_axisbelow(True)
    for sp in ("top", "right"):
        b.spines[sp].set_visible(False)


def finish_maps(fig, axes, cbars, title, mono, muted):
    """Place colourbars, headline and footers against the maps' real edges.
    Map axes keep their projection aspect, so a fixed suptitle position
    leaves dead space above them; draw first, then measure.
    cbars: list of (mappable, axes it spans, label)."""
    fig.canvas.draw()                              # applies the map aspect
    H = fig.get_figheight()
    pos = {b: b.get_position() for b in axes}
    y_top = max(p.y1 for p in pos.values())
    y_bot = min(p.y0 for p in pos.values())
    cb_h, cb_gap = 0.15 / H, 0.12 / H
    y_cb = y_bot - cb_gap - cb_h
    for im, span, label in cbars:
        x0 = min(pos[b].x0 for b in span)
        x1 = max(pos[b].x1 for b in span)
        wd = x1 - x0
        cax = fig.add_axes([x0 + 0.15 * wd, y_cb, 0.70 * wd, cb_h])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                          extend="both")
        cb.set_label(label, fontsize=10)
        cb.outline.set_edgecolor(INK2)
    fig.suptitle(title, x=0.012, ha="left", va="bottom", y=y_top + 0.45 / H,
                 fontsize=13.5, fontweight="bold")
    # footers stack downward; labels and footers may wrap onto several lines
    nl = lambda t: t.count("\n") + 1
    lab_lines = max(nl(lbl) for _, _, lbl in cbars)
    y_mono = y_cb - (0.50 + 0.17 * lab_lines) / H
    fig.text(0.012, y_mono, mono, fontsize=8.8, color=INK2,
             family="monospace", va="top", linespacing=1.4)
    fig.text(0.012, y_mono - (0.17 * nl(mono) + 0.08) / H, muted,
             fontsize=8.2, color=MUTED, va="top", linespacing=1.4)


def mark_readout(b, readout, data):
    name, rlat, rlon = readout
    b.plot(rlon, rlat, marker="*", ms=15, color="white", mec=INK, mew=1.2,
           transform=data, zorder=5)
    b.text(rlon + 6, rlat + 4, name, fontsize=10, fontweight="bold",
           color=INK, transform=data, zorder=5)


def save(fig, out):
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight",
                    pad_inches=0.25)
    plt.close(fig)
    print(f"    wrote {out}.pdf / .png")


# ------------------------------------------------- figure A: velocity maps
def figure_A(depth_km, lat, lon, s2d, p2d, vmax, readout, out):
    """dlnVp (top) and dlnVs (bottom) on ONE colour scale. On a shared scale
    dlnVs ~ 2x dlnVp shows directly as the bottom panel being more saturated:
    that contrast is R."""
    import cartopy.crs as ccrs

    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], s2d.shape)
    rms_p = float(np.sqrt((w * p2d ** 2).sum() / w.sum()))
    rms_s = float(np.sqrt((w * s2d ** 2).sum() / w.sum()))
    clip_p = float((w * (np.abs(p2d) > vmax)).sum() / w.sum())
    clip_s = float((w * (np.abs(s2d) > vmax)).sum() / w.sum())
    print(f"\n  figure A, {depth_km:.0f} km:  RMS dlnVp = {rms_p:.3f}%  "
          f"RMS dlnVs = {rms_s:.3f}%  R = {rms_s / rms_p:.2f}")
    print(f"    shared scale +/-{vmax:g}%: clipped area dlnVp {clip_p:.1%}, "
          f"dlnVs {clip_s:.1%}")
    print("    VERIFY: top (dlnVp) panel paler than bottom (dlnVs).")

    proj, data = ccrs.Robinson(), ccrs.PlateCarree()
    # stacked; height chosen so each Robinson map (aspect ~2:1) fills the
    # width, with room between them for the lower panel's title
    fig, ax = plt.subplots(2, 1, figsize=(7.5, 8.4),
                           subplot_kw={"projection": proj})
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02,
                        hspace=0.16)
    for b, field, name, lab in [(ax[0], p2d, "dlnVp", "a"),
                                (ax[1], s2d, "dlnVs", "b")]:
        im = b.pcolormesh(lon, lat, field, cmap="RdBu", vmin=-vmax, vmax=vmax,
                          shading="nearest", transform=data, rasterized=True)
        b.coastlines(lw=0.6, color=INK)
        b.set_global()
        mark_readout(b, readout, data)
        b.set_title(f"{lab}  {name} at {depth_km:.0f} km", loc="left",
                    fontweight="bold", fontsize=12.5, pad=10)

    finish_maps(
        fig, list(ax),
        [(im, list(ax), "velocity anomaly relative to PREM (%), "
                        "same scale in both panels\n"
                        "red = slow, blue = fast")],
        f"On one colour scale, S anomalies are {rms_s / rms_p:.1f}× "
        f"stronger than P",
        f"SP12RTS {depth_km:.0f} km  |  full grid ({s2d.size} pts)  |  "
        f"colour scale ±{vmax:g}% both panels\n"
        f"RMS dlnVp={rms_p:.2f}%  dlnVs={rms_s:.2f}%  R={rms_s / rms_p:.2f}",
        "SP12RTS (Koelemeijer et al. 2016), percent relative to PREM.  "
        "RMS area-weighted by cos(latitude).\n"
        f"{Path(out).name}  {datetime.now():%Y-%m-%d %H:%M}")
    save(fig, out)


# ----------------------------------------------------- PINN .npz helpers
def find_npz():
    runs = Path(__file__).resolve().parent / "runs"       # PINN default
    hits = sorted(runs.glob("pinn_d500_*_w_base.npz"))
    return str(hits[-1]) if hits else None


def load_pinn(path):
    z = np.load(path, allow_pickle=False)
    need = ["x_E1", "x_E2", "y_E2", "x_true", "y_true", "p_obs", "s_obs",
            "lat", "lon", "readout_index", "vs0", "arg_rt", "arg_rc"]
    miss = [k for k in need if k not in z.files]
    if miss:
        raise SystemExit(f"{path} lacks {miss}; rerun pinn_degeneracy.py")
    d = {k: z[k] for k in z.files}
    # the network's own y must be stored. If y_E2 is exactly p_obs - x_E2
    # it was derived from x, which forces corr(x, y) = -1 by construction.
    if np.allclose(d["y_E2"], d["p_obs"][None, :] - d["x_E2"], rtol=0, atol=0):
        raise SystemExit(f"{path}: y_E2 == p_obs - x_E2 exactly - this .npz "
                         "stores derived y, not the network output. Rerun.")
    d["rt"], d["rc"] = float(d["arg_rt"]), float(d["arg_rc"])
    d["depth"] = float(d["depth"])
    d["k"] = int(d["readout_index"])
    d["name"] = str(d.get("arg_readout_name", "CVL"))
    return d


def to_T(d, x):
    """dT (K) from the thermal share x (% of dlnVp), as in the PINN."""
    return x / 100.0 * d["rt"] * float(d["vs0"]) / DVS_DT


def to_Mg(y):
    """dMg# from the compositional share y (% of dlnVp), as in the PINN."""
    return MG_PER_Y * y


# ---------------------------------------------- figure B: per-seed figure
def figure_B(d, path, out):
    k, name = d["k"], d["name"]
    runs = [("P only", d["x_E1"], ORANGE), ("P + S", d["x_E2"], BLUE)]
    T_exact = float(to_T(d, d["x_true"][k]))

    print(f"\n  figure B from {Path(path).name}")
    print(f"    {name}: algebraic solution dT = {T_exact:+.1f} K")
    spread = {}
    for lab, x, _ in runs:
        T = to_T(d, x[:, k])
        spread[lab] = float(np.median(to_T(d, x).std(0)))
        print(f"    {lab:7s}: dT at {name} = {T.mean():+7.1f} +/- {T.std():.1f} K"
              f"   median per-point spread = {spread[lab]:.1f} K")
    print("    VERIFY: P + S ~ +108.5 +/- 3.1 K, P only spread ~ 30 K, "
          "median spreads 31.9 K and 3.9 K.")

    fig, ax = plt.subplots(1, 2, figsize=(13.0, 6.4))
    rng = np.random.default_rng(1)
    for j, (lab, x, col) in enumerate(runs):
        T = to_T(d, x[:, k])
        jit = (rng.random(T.size) - 0.5) * 0.26
        ax[0].scatter(np.full(T.size, j) + jit, T, s=80, color=col,
                      edgecolor="white", lw=1.4, zorder=4)
    ax[0].axhline(T_exact, color=INK, lw=1.8, ls=(0, (5, 3)), zorder=3)
    ax[0].text(-0.46, T_exact + 2.0, "algebraic solution of P + S",
               fontsize=10, color=INK, va="bottom", ha="left",
               fontweight="bold")
    ax[0].set_xticks([0, 1])
    ax[0].set_xticklabels(["trained on\n$V_P$ only",
                           "trained on\n$V_P$ + $V_S$"])
    ax[0].set_xlim(-0.5, 1.5)
    ax[0].set_ylabel(f"recovered ΔT at {name} (K)", fontweight="bold")
    ax[0].set_title(f"a  {d['x_E1'].shape[0]} seeds, same data", loc="left",
                    fontweight="bold", fontsize=12.5, pad=12)

    for lab, x, col in runs:
        ax[1].hist(to_T(d, x).std(0), bins=60, color=col, alpha=0.65,
                   label=lab, zorder=3)
    ax[1].set_xlabel("seed-to-seed spread in ΔT (K), per grid point",
                     fontweight="bold")
    ax[1].set_ylabel("number of grid points", fontweight="bold")
    ax[1].set_yscale("log")
    ax[1].legend(frameon=False, fontsize=10.5)
    ax[1].set_title("b  whole map, per grid point", loc="left",
                    fontweight="bold", fontsize=12.5, pad=12)
    for b in ax:
        style(b)

    fig.suptitle("Fitting P alone leaves temperature undetermined; "
                 "adding S pins it down",
                 x=0.012, ha="left", fontsize=13.5, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0.075, 1, 0.95))
    fig.text(0.012, 0.042,
             f"SP12RTS {d['depth']:.0f} km  |  {d['p_obs'].size} pts  |  "
             f"seeds={d['x_E1'].shape[0]}  |  R_T={d['rt']:g}  "
             f"R_C={d['rc']:g}  |  median spread {spread['P only']:.1f} K "
             f"-> {spread['P + S']:.1f} K",
             fontsize=8.8, color=INK2, family="monospace")
    fig.text(0.012, 0.012,
             f"ΔT from dVs/dT = {DVS_DT} m/s/K (Faul & Jackson 2005), "
             f"Vs0 = {float(d['vs0']):.0f} m/s (PREM).  "
             f"R_T, R_C from Cammarano et al. (2003).  {Path(path).stem}",
             fontsize=8.2, color=MUTED)
    save(fig, out)


# ------------------------------- figure C: recovered fields, masked by R
def figure_C(d, path, out):
    """Mean P + S recovered dT and dMg#, greyed wherever the point ratio
    dlnVs/dlnVp is outside [R_C, R_T]: no mix of the two mechanisms can
    produce it there."""
    import cartopy.crs as ccrs

    lat, lon, k, name = d["lat"], d["lon"], d["k"], d["name"]
    shape = (lat.size, lon.size)
    p, s = d["p_obs"], d["s_obs"]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = s / p                                  # p = 0 -> inf/nan -> grey
    inside = (r >= d["rc"]) & (r <= d["rt"])       # coloured where True
    w = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], shape).ravel()
    frac = float((w * inside).sum() / w.sum())

    T = to_T(d, d["x_E2"].mean(0))
    Mg = to_Mg(d["y_E2"].mean(0))                  # network's own y
    ilat, ilon = divmod(k, lon.size)
    print(f"\n  figure C from {Path(path).name}")
    print(f"    area-weighted coloured fraction (R_C <= R <= R_T) = {frac:.1%}"
          f"   (grey {1 - frac:.1%})")
    print(f"    {name} ({lat[ilat]:.0f}, {lon[ilon]:.0f}): R = {r[k]:.2f}  "
          f"{'COLOURED' if inside[k] else 'grey'}   dT = {T[k]:+.1f} K  "
          f"dMg# = {Mg[k]:+.2f}")
    print("    VERIFY: coloured ~11%; more than ~20% means the mask is "
          f"inverted; {name} must be grey.")

    proj, data = ccrs.Robinson(), ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 2, figsize=(14.0, 3.7),
                           subplot_kw={"projection": proj})
    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.03,
                        wspace=0.04)
    cbars = []
    for b, field, lab, title, cmap, unit in [
            (ax[0], T, "a", "recovered ΔT", "RdBu_r", "ΔT (K)"),
            (ax[1], Mg, "b", "recovered ΔMg# (Fe-equivalent)", "PuOr",
             "ΔMg#  (negative = Fe-rich)")]:
        lim = float(np.percentile(np.abs(field[inside]), 98)) if inside.any() \
            else float(np.abs(field).max())
        b.pcolormesh(lon, lat, np.ones(shape), cmap=matplotlib.colors
                     .ListedColormap([GREY]), shading="nearest",
                     transform=data, rasterized=True)
        im = b.pcolormesh(lon, lat,
                          np.ma.masked_where(~inside, field).reshape(shape),
                          cmap=cmap, vmin=-lim, vmax=lim, shading="nearest",
                          transform=data, rasterized=True)
        b.coastlines(lw=0.6, color=INK)
        b.set_global()
        mark_readout(b, (name, lat[ilat], lon[ilon]), data)
        b.set_title(f"{lab}  {title}, P + S", loc="left", fontweight="bold",
                    fontsize=12.5, pad=10)
        cbars.append((im, [b], unit))

    finish_maps(
        fig, list(ax), cbars,
        f"Where two end-members suffice is a contour, not a region "
        f"({d['depth']:.0f} km)",
        f"grey: observed dlnVs/dlnVp outside [R_C, R_T] = "
        f"[{d['rc']:g}, {d['rt']:g}]  |  coloured {frac:.1%} of area  |  "
        f"mean of {d['x_E2'].shape[0]} seeds",
        f"ΔT from dVs/dT = {DVS_DT} m/s/K (Faul & Jackson 2005); "
        f"ΔMg# = {MG_PER_Y} × y, R_T, R_C from Cammarano et al. (2003).  "
        f"Area-weighted by cos(latitude).  {Path(path).stem}")
    save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vs_file")
    ap.add_argument("vp_file")
    ap.add_argument("--out", default="sp12rts_eval")
    ap.add_argument("--mask-above", type=float, default=None,
                    help="set R to NaN for depths shallower than this (km)")
    ap.add_argument("--depths", type=float, nargs="*", default=[500.0, 700.0],
                    help="depths (km) for figure A; each snaps to the nearest "
                         "model level")
    ap.add_argument("--vmax", type=float, default=1.2,
                    help="shared colour limit (%%) for both panels of figure A")
    ap.add_argument("--npz", default=None,
                    help="weighted PINN run for figures B and C (default: "
                         "newest runs/pinn_d500_*_w_base.npz)")
    a = ap.parse_args()

    ds_s = xr.open_dataset(a.vs_file)
    ds_p = xr.open_dataset(a.vp_file)
    vs = ds_s["v"]
    vp = ds_p["v"]

    print("INPUT")
    print(f"  Vs: {vs.shape}  units={vs.attrs.get('units','?')}")
    print(f"  Vp: {vp.shape}  units={vp.attrs.get('units','?')}")

    depth = ds_s["depth"].values.astype(float)
    lat = ds_s["latitude"].values.astype(float)
    assert np.allclose(depth, ds_p["depth"].values), "depth grids differ"

    # percent -> fraction
    S = vs.values / 100.0
    P = vp.values / 100.0

    w = area_weights(lat)[None, :, None]          # broadcast over (dep,lat,lon)
    W = np.broadcast_to(w, S.shape)

    def wmean(x):
        return (x * W).sum(axis=(1, 2)) / W.sum(axis=(1, 2))

    rms_s = np.sqrt(wmean(S ** 2))
    rms_p = np.sqrt(wmean(P ** 2))
    R_rms = rms_s / rms_p

    # bulk sound speed
    vp0 = np.interp(depth, PREM_DEPTH, PREM_VP)
    vs0 = np.interp(depth, PREM_DEPTH, PREM_VS)
    vc0sq = vp0 ** 2 - (4.0 / 3.0) * vs0 ** 2
    cp = (vp0 ** 2 / vc0sq)[:, None, None]
    cs = ((4.0 / 3.0) * vs0 ** 2 / vc0sq)[:, None, None]
    C = cp * P - cs * S

    n = depth.size
    R_med = np.full(n, np.nan)
    R_b1 = np.full(n, np.nan)                      # Vp exact (lower bound)
    R_b2 = np.full(n, np.nan)                      # Vs exact (upper bound)
    r0 = np.full(n, np.nan)                        # uncentred correlation
    corr_sp = np.full(n, np.nan)
    corr_sc = np.full(n, np.nan)
    THRESH = 0.001                                 # 0.1 per cent

    for i in range(n):
        s, p, c = S[i].ravel(), P[i].ravel(), C[i].ravel()
        ww = np.broadcast_to(w[0], S[i].shape).ravel()

        def wcorr(x, y):
            mx = (ww * x).sum() / ww.sum()
            my = (ww * y).sum() / ww.sum()
            cx, cy = x - mx, y - my
            num = (ww * cx * cy).sum()
            den = np.sqrt((ww * cx ** 2).sum() * (ww * cy ** 2).sum())
            return num / den if den > 0 else np.nan

        corr_sp[i] = wcorr(s, p)
        corr_sc[i] = wcorr(s, c)
        ssp, spp, sss = ((ww * s * p).sum(), (ww * p ** 2).sum(),
                         (ww * s ** 2).sum())
        R_b1[i] = ssp / spp                        # Vp exact (lower)
        R_b2[i] = sss / ssp                        # Vs exact (upper)
        r0[i] = ssp / np.sqrt(sss * spp)

        m = (np.abs(s) >= THRESH) & (np.abs(p) >= THRESH)
        if m.sum() > 10:
            R_med[i] = np.median(s[m] / p[m])

    # ------------------------------------------------------------ printout
    # every level down to 1000 km (so the contaminated top layers are
    # visible for choosing --mask-above), then sparser below
    print("\n  depth   RMS dVs%  RMS dVp%   R(rms)  R(med)   R_b1   R_b2"
          "   r(s,p)   r(s,c)")
    print("-" * 84)
    step = max(1, n // 30)
    rows = sorted(set(np.flatnonzero(depth <= 1000))
                  | set(range(0, n, step)) | {n - 1})
    for i in rows:
        print(f"{depth[i]:7.0f} {100*rms_s[i]:9.3f} {100*rms_p[i]:9.3f} "
              f"{R_rms[i]:8.2f} {R_med[i]:7.2f} {R_b1[i]:6.2f} {R_b2[i]:6.2f} "
              f"{corr_sp[i]:8.2f} {corr_sc[i]:8.2f}")

    ratio = R_b2 / R_b1
    print(f"\n  bracket identity: max |R_b2/R_b1 - 1/r0^2| = "
          f"{np.nanmax(np.abs(ratio - 1 / r0 ** 2)):.1e}   "
          f"(vs centred r(s,p): {np.nanmax(np.abs(ratio - 1 / corr_sp ** 2)):.1e})")
    print(f"  geometric mean:   max |sqrt(R_b1*R_b2) - R_rms| = "
          f"{np.nanmax(np.abs(np.sqrt(R_b1 * R_b2) - R_rms)):.1e}")

    # ------------------------------------------------- mask the top layers
    if a.mask_above is not None:
        top = depth < a.mask_above
        for arr in (R_rms, R_med, R_b1, R_b2):
            arr[top] = np.nan
        print(f"\n  masked R at {top.sum()} levels shallower than "
              f"{a.mask_above:g} km (Vp<->Vs projection, Koelemeijer et al. 2016)")

    # ------------------------------------------------------------- verdict
    deep = depth >= 1000
    pk = np.argmax(np.where(deep & np.isfinite(R_rms), R_rms, -np.inf))
    lower = depth >= 2550
    lm = depth >= 1500

    checks = [
        ("peak |dlnVs| near 3.5% (scaling)",
         2.5 <= np.abs(S).max() * 100 <= 4.5,
         f"max |dlnVs| = {np.abs(S).max()*100:.2f}%"),
        ("dlnVs RMS exceeds dlnVp RMS at all depths",
         bool((rms_s > rms_p).all()),
         f"min ratio = {(rms_s / rms_p).min():.2f}"),
        ("R peaks between 2300-2700 km",
         2300 <= depth[pk] <= 2700,
         f"peak at {depth[pk]:.0f} km"),
        ("peak R in published range 2.0-6.0",
         2.0 <= R_rms[pk] <= 6.0,
         f"R(rms)={R_rms[pk]:.2f}  R(med)={R_med[pk]:.2f}  "
         f"R_b=[{R_b1[pk]:.2f}, {R_b2[pk]:.2f}]"),
        ("R decreases through lowermost ~300 km (D'')",
         R_rms[-1] < R_rms[pk],
         f"{R_rms[pk]:.2f} -> {R_rms[-1]:.2f}"),
        ("dlnVs-dlnVp correlation positive at all depths",
         bool((corr_sp > 0).all()),
         f"min r = {np.nanmin(corr_sp):.2f}"),
        ("dlnVs-dlnVc anticorrelated in lower mantle (>1500 km)",
         bool(np.nanmean(corr_sc[lm]) < 0),
         f"mean r(s,c) below 1500 km = {np.nanmean(corr_sc[lm]):.2f}"),
    ]
    print("\nVERDICT  (Koelemeijer et al. 2016 signature)")
    print("=" * 76)
    for lab, ok, det in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {lab:<52} {det}")
    nfail = sum(1 for _, o, _ in checks if not o)
    print(f"\n  {len(checks)-nfail}/{len(checks)} passed.")

    # ---------------------------------------------------------------- plot
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 7.2), sharey=True)

    ax[0].plot(100 * rms_s, depth, lw=2.2, color=ORANGE, label="dlnVs")
    ax[0].plot(100 * rms_p, depth, lw=2.2, color=BLUE, label="dlnVp")
    ax[0].set_xlabel("Area-weighted RMS (%)", fontweight="bold")
    ax[0].set_ylabel("Depth (km)", fontweight="bold")
    ax[0].legend(frameon=False)
    ax[0].set_title("a  Heterogeneity amplitude", loc="left",
                    fontweight="bold", fontsize=12)

    ax[1].fill_betweenx(depth, R_b1, R_b2, color=BLUE, alpha=0.22, lw=0,
                        label=r"regression bracket (width $1/r^2$)")
    ax[1].plot(R_rms, depth, lw=2.2, color=INK, label="RMS ratio")
    ax[1].plot(R_med, depth, lw=1.8, color=ORANGE, ls="--", label="median")
    ax[1].axvline(2.7, color=MUTED, lw=1, ls=":")
    ax[1].axhline(depth[pk], color=MUTED, lw=1, ls=":")
    ax[1].annotate(f"peak {depth[pk]:.0f} km", xy=(0.06, depth[pk] - 60),
                   fontsize=9.5, color=INK2)
    ax[1].set_xlabel(r"$R = d\ln V_S / d\ln V_P$", fontweight="bold")
    ax[1].set_xlim(0, 6)
    ax[1].legend(frameon=False, loc="upper right")
    ax[1].set_title("b  Velocity ratio", loc="left",
                    fontweight="bold", fontsize=12)

    ax[2].plot(corr_sp, depth, lw=2.2, color=INK,
               label=r"$r(d\ln V_S,\ d\ln V_P)$")
    ax[2].plot(corr_sc, depth, lw=2.2, color=ORANGE,
               label=r"$r(d\ln V_S,\ d\ln V_C)$")
    ax[2].axvline(0, color=MUTED, lw=1)
    ax[2].set_xlabel("Correlation", fontweight="bold")
    ax[2].set_xlim(-1, 1)
    ax[2].legend(frameon=False, loc="lower left", fontsize=9.5)
    ax[2].set_title("c  Thermal vs. chemical signature", loc="left",
                    fontweight="bold", fontsize=12)

    for b in ax:
        b.invert_yaxis()
        b.set_ylim(2950, 0)
        b.grid(color=GRID, lw=0.7)
        b.set_axisbelow(True)
        for s_ in ("top", "right"):
            b.spines[s_].set_visible(False)

    fig.suptitle("SP12RTS verification and velocity-ratio diagnostics",
                 x=0.012, ha="left", fontsize=13.5, fontweight="bold", y=0.985)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    note = ("SP12RTS (Koelemeijer et al. 2016, GJI 204, 1024), percent "
            "relative to PREM.  Median R masks points with |dlnV| < 0.1% "
            "following the original.  Bulk sound dlnVc from PREM Vp, Vs.\n"
            "Band: regression of dlnVs on dlnVp (lower) and dlnVp on dlnVs "
            "(upper); R_rms is their geometric mean.")
    if a.mask_above is not None:
        note += (f"  R not shown above {a.mask_above:g} km: Vp-Vs projection "
                 "in the top spline layer (Koelemeijer et al. 2016).")
    fig.text(0.012, 0.004, note, fontsize=8.4, color=MUTED)

    fig.savefig(f"{a.out}.pdf", bbox_inches="tight", pad_inches=0.25)
    fig.savefig(f"{a.out}.png", dpi=200, bbox_inches="tight", pad_inches=0.25)
    np.savez_compressed(f"{a.out}.npz", depth=depth, rms_vs=rms_s,
                        rms_vp=rms_p, R_rms=R_rms, R_med=R_med, R_b1=R_b1,
                        R_b2=R_b2, r0=r0, corr_sp=corr_sp, corr_sc=corr_sc,
                        mask_above=np.nan if a.mask_above is None
                        else a.mask_above)
    print(f"\n  wrote {a.out}.pdf / .png / .npz")

    # PINN run first: its readout point is the CVL marked on figure A
    npz = a.npz or find_npz()
    pinn = load_pinn(npz) if npz else None
    if pinn is not None:
        ilat, ilon = divmod(pinn["k"], pinn["lon"].size)
        readout = (pinn["name"], float(pinn["lat"][ilat]),
                   float(pinn["lon"][ilon]))
    else:
        readout = ("CVL", 0.0, 12.0)    # PINN default 3N 10E on its sub=6 grid

    # ------------------------------------------- figure A: velocity maps
    lon = ds_s["longitude"].values.astype(float)
    for d in a.depths:
        i = int(np.argmin(np.abs(depth - d)))
        figure_A(depth[i], lat, lon, 100 * S[i], 100 * P[i], a.vmax, readout,
                 f"{a.out}_mapsA_d{depth[i]:.0f}")

    # ------------------------------------- figures B and C: PINN results
    if pinn is None:
        print("\n  no weighted PINN .npz found; skipping figures B and C "
              "(pass --npz)")
        return
    figure_B(pinn, npz, f"{a.out}_seedsB")
    figure_C(pinn, npz, f"{a.out}_fieldsC")


if __name__ == "__main__":
    main()
