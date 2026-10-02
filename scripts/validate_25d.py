"""Phase 5 validation of the 2.5D input pipeline on the cached patches (data/processed).

A. Integrity + slice order: re-extract patches with an INDEPENDENT indexing path from freshly
   preprocessed scans and require bit-exact equality with the cache; a reversed-order control
   must differ (proves the check is sensitive to ordering).
B. Boundary handling: candidates within 2 slices of the scan ends (counts + exactness incl. edge replication).
C. Annotation alignment: for every positive in the cache, box offset / size / contrast / z-offset.
D. 2D vs 2.5D: how much information the extra slices add (inter-slice correlation, difference,
   nodule profile along z by nodule size).
Outputs: data/metadata/25d_validation.json and results/figures/exp000_25d_*.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing.pipeline import preprocess_scan
from src.utils.config import load_config
from src.utils.progress import Progress

DATA, PROC, FIG = Path("data"), Path("data/processed"), Path("results/figures")
C, P, H = 5, 64, 2      # channels, patch size, half window
rng = np.random.default_rng(0)


def load(split):
    return np.load(PROC / f"patches_{split}.npy", mmap_mode="r"), pd.read_csv(PROC / f"patches_{split}.csv")


def reference_patch(vol, w, depth_half=H):
    """Independent re-implementation: pad-then-crop, clamped slice indices."""
    x, y, z = (int(round(v)) for v in vol.world_to_voxel(w))
    D = vol.array.shape[0]
    if not 0 <= z < D:
        return np.zeros((2 * depth_half + 1, P, P), np.float32)
    zs = [min(max(z + k, 0), D - 1) for k in range(-depth_half, depth_half + 1)]
    pad = np.pad(vol.array[zs], ((0, 0), (P // 2, P // 2), (P // 2, P // 2)), constant_values=0.0)
    return pad[:, y:y + P, x:x + P]       # centre pixel at index P/2, i.e. rows y-32 .. y+31


def check_integrity_and_boundary(scans, cidx, cfg):
    stores = {s: load(s) for s in ("train", "val", "test")}
    cidx = cidx.rename(columns={"class": "label"})
    cidx["z_res"] = cidx.vox_z * cidx.seriesuid.map(scans.set_index("seriesuid").sp_z)
    cidx["depth_res"] = np.round(cidx.seriesuid.map(scans.set_index("seriesuid").eval("n_slices * sp_z")))
    cidx["near_edge"] = (np.round(cidx.z_res) < H) | (np.round(cidx.z_res) > cidx.depth_res - 1 - H)
    stats = {"candidates_near_scan_end": {s: int(cidx[(cidx.split == s)].near_edge.sum()) for s in stores},
             "positives_near_scan_end": int(cidx[(cidx.label == 1) & cidx.near_edge].shape[0])}
    # sample: scans containing boundary candidates + random scans
    keyed = {s: m.assign(row=np.arange(len(m))).set_index(["seriesuid", "coordX", "coordY", "coordZ"]) for s, (_, m) in stores.items()}
    pick = []
    edge_scans = cidx[cidx.near_edge].seriesuid.unique()
    for uid in list(rng.choice(edge_scans, min(6, len(edge_scans)), replace=False)) + list(rng.choice(scans.seriesuid, 8, replace=False)):
        g = cidx[(cidx.seriesuid == uid)]
        s = g.split.iloc[0]
        rows = [g[g.label == 1], g[g.near_edge].head(6), g.sample(min(5, len(g)), random_state=1)]
        for r in pd.concat(rows).drop_duplicates(["coordX", "coordY", "coordZ"]).itertuples():
            k = (r.seriesuid, r.coordX, r.coordY, r.coordZ)
            if k in keyed[s].index:           # train cache holds only a sample of negatives
                pick.append((s, uid, int(keyed[s].loc[k, "row"]), r.near_edge, (r.coordX, r.coordY, r.coordZ)))
    n_ok = n_rev_differs = n_edge = n_edge_ok = 0
    mism = []
    by_scan = {}
    for t in pick:
        by_scan.setdefault((t[1], t[0]), []).append(t)
    sub = scans.set_index("seriesuid").subset
    for (uid, s), items in Progress(list(by_scan.items()), desc="integrity check", unit="scan", step_pct=20):
        vol = preprocess_scan(DATA, int(sub[uid]), uid, cfg).volume
        for _, _, row, edge, w in items:
            cached = stores[s][0][row]
            ref = np.rint(reference_patch(vol, w) * 255).astype(np.uint8)
            ok = np.array_equal(cached, ref)
            n_ok += ok
            n_rev_differs += not np.array_equal(cached, ref[::-1])
            if edge:
                n_edge += 1
                n_edge_ok += ok
            if not ok:
                mism.append((uid[-8:], int(row)))
    stats.update({"integrity_checked": len(pick), "integrity_exact_match": int(n_ok),
                  "reversed_order_differs_from_cache": int(n_rev_differs), "boundary_checked": n_edge,
                  "boundary_exact_match": int(n_edge_ok), "mismatches": mism[:10]})
    return stats


def check_all_boundary_candidates(scans, cidx, cfg):
    """Fresh extraction (not via the cache) for EVERY candidate within 2 slices of a scan end:
    must equal the independent reference, and out-of-range channels must replicate the edge slice."""
    from src.preprocessing.patches import extract_25d_patch
    sc = scans.set_index("seriesuid")
    z = cidx.vox_z * cidx.seriesuid.map(sc.sp_z)
    depth = np.round(cidx.seriesuid.map(sc.n_slices * sc.sp_z))
    e = cidx[(np.round(z) < H) | (np.round(z) > depth - 1 - H)]
    ok = rep_ok = n = 0
    for uid, g in Progress(list(e.groupby("seriesuid")), desc="boundary candidates", unit="scan", step_pct=25):
        vol = preprocess_scan(DATA, int(sc.loc[uid, "subset"]), uid, cfg).volume
        D = vol.array.shape[0]
        for r in g.itertuples():
            w = (r.coordX, r.coordY, r.coordZ)
            got, c = extract_25d_patch(vol, w, P, C, "edge", 0.0, 1)
            ok += np.allclose(got, reference_patch(vol, w))
            zc = int(round(c[2]))
            lo, hi = (got[: H - zc] if zc < H else None), (got[H + (D - 1 - zc) + 1:] if zc > D - 1 - H else None)
            checks = [np.array_equal(x, np.broadcast_to(got[[H - zc if zc < H else H + (D - 1 - zc)][0]], x.shape)) for x in (lo, hi) if x is not None and len(x)]
            rep_ok += all(checks)
            n += 1
    return {"candidates_checked": n, "match_independent_reference": int(ok), "edge_replication_correct": int(rep_ok)}


def check_alignment_and_compare(scans, ann):
    out, rows, patches = {}, [], []
    for s in ("train", "val", "test"):
        arr, m = load(s)
        pos = m.index[m.label == 1].to_numpy()
        patches.append(np.asarray(arr[pos]))
        rows.append(m.loc[pos].assign(split=s))
    pos_x = np.concatenate(patches)                          # (Npos, 5, 64, 64) uint8
    pm = pd.concat(rows, ignore_index=True)
    # matched nodule (nearest annotation in the same scan) -> z offset
    dz = []
    for r in pm.itertuples():
        a = ann[ann.seriesuid == r.seriesuid]
        d = np.linalg.norm(a[["coordX", "coordY", "coordZ"]].to_numpy() - np.array([r.coordX, r.coordY, r.coordZ]), axis=1)
        j = d.argmin()
        dz.append(float(a.iloc[j].coordZ - r.coordZ))        # mm == slices at 1 mm spacing
    pm["dz"] = dz
    xy = np.hypot(pm.box_dx, pm.box_dy)
    out["positives"] = int(len(pm))
    out["boxes_present_frac"] = float(pm.box_d.notna().mean())
    out["box_offset_px"] = {"median": float(xy.median()), "p95": float(xy.quantile(.95)), "max": float(xy.max())}
    out["box_inside_patch_frac"] = float(((pm.box_dx.abs() + pm.box_d / 2 <= P / 2) & (pm.box_dy.abs() + pm.box_d / 2 <= P / 2)).mean())
    out["nodule_centre_within_window_frac"] = float((pm.dz.abs() <= H + 0.5).mean())
    out["abs_dz_mm"] = {"median": float(pm.dz.abs().median()), "p95": float(pm.dz.abs().quantile(.95)), "max": float(pm.dz.abs().max())}

    # contrast inside box vs surrounding ring on the centre channel
    contrast = []
    yy, xx = np.mgrid[:P, :P]
    for i, r in enumerate(pm.itertuples()):
        cx, cy, rad = P / 2 + r.box_dx, P / 2 + r.box_dy, r.box_d / 2
        d = np.hypot(xx - cx, yy - cy)
        img = pos_x[i, H].astype(np.float32) / 255
        ring = (d > rad + 2) & (d <= rad + 6)
        contrast.append(img[d <= max(rad * .7, 1)].mean() - img[ring].mean() if ring.any() else np.nan)
    contrast = np.array(contrast)
    out["centre_slice_contrast_nodule_minus_ring"] = {"median": float(np.nanmedian(contrast)), "frac_positive": float(np.nanmean(contrast > 0))}

    # nodule intensity profile along z (normalised by centre channel), by size
    size_bins = [("small <6 mm", pm.box_d < 6), ("medium 6-10 mm", (pm.box_d >= 6) & (pm.box_d < 10)), ("large >=10 mm", pm.box_d >= 10)]
    prof = {}
    for name, mask in size_bins:
        vals = np.zeros((int(mask.sum()), C))
        for j, i in enumerate(np.where(mask)[0]):
            r = pm.iloc[i]
            d = np.hypot(xx - (P / 2 + r.box_dx), yy - (P / 2 + r.box_dy))
            reg = d <= max(r.box_d / 2 * .7, 1)
            vals[j] = [pos_x[i, c][reg].mean() / 255 for c in range(C)]
        prof[name] = {"n": int(mask.sum()), "mean_profile": (vals.mean(0)).round(4).tolist()}
    out["nodule_profile_by_size"] = prof

    # 2D vs 2.5D: how different are the extra channels? positives vs random negatives
    arr, m = load("train")
    neg_idx = np.sort(rng.choice(np.where(m.label == 0)[0], 3000, replace=False))
    neg_x = np.asarray(arr[neg_idx])
    def chan_stats(x):
        f = x.astype(np.float32) / 255
        ctr = f[:, H].reshape(len(f), -1)
        corr, mad = [], []
        for c in range(C):
            o = f[:, c].reshape(len(f), -1)
            a, b = o - o.mean(1, keepdims=True), ctr - ctr.mean(1, keepdims=True)
            corr.append(float(((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)).mean()))
            mad.append(float(np.abs(o - ctr).mean()))
        return corr, mad
    pc, pmad = chan_stats(pos_x)
    nc, nmad = chan_stats(neg_x)
    out["corr_with_centre_slice"] = {"positives": np.round(pc, 4).tolist(), "negatives": np.round(nc, 4).tolist()}
    out["mean_abs_diff_to_centre"] = {"positives": np.round(pmad, 4).tolist(), "negatives": np.round(nmad, 4).tolist()}
    return out, pos_x, pm, neg_x, prof, (pc, nc, pmad, nmad)


def figures(pos_x, pm, neg_x, prof, cs):
    FIG.mkdir(parents=True, exist_ok=True)
    # 1. samples
    rowspec = []
    for name, mask in (("small", pm.box_d < 6), ("medium", (pm.box_d >= 6) & (pm.box_d < 10)), ("large", pm.box_d >= 10)):
        for i in rng.choice(np.where(mask)[0], 2, replace=False):
            rowspec.append((f"nodule {name} (d={pm.box_d.iloc[i]:.1f} mm)", pos_x[i], pm.iloc[i]))
    for i in range(2):
        rowspec.append(("random candidate (negative)", neg_x[i], None))
    fig, axes = plt.subplots(len(rowspec), C, figsize=(1.9 * C, 1.95 * len(rowspec)))
    for r, (title, patch, meta) in enumerate(rowspec):
        for c in range(C):
            ax = axes[r, c]
            ax.imshow(patch[c], cmap="gray", vmin=0, vmax=255); ax.axis("off")
            if meta is not None:
                ax.add_patch(plt.Rectangle((P / 2 + meta.box_dx - meta.box_d / 2, P / 2 + meta.box_dy - meta.box_d / 2),
                                           meta.box_d, meta.box_d, fill=False, ec="#DC2626", lw=1.1))
            if r == 0:
                ax.set_title("z" if c == H else f"z{c - H:+d}", fontsize=9)
        axes[r, 0].text(-0.05, 0.5, title.replace(" (", "\n("), transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=7)
    plt.tight_layout(); plt.subplots_adjust(left=0.2); plt.savefig(FIG / "exp000_25d_samples.png", dpi=130); plt.close()
    # 2. difference maps
    fig, axes = plt.subplots(2, 4, figsize=(9.5, 5))
    for k, i in enumerate(rng.choice(len(pos_x), 4, replace=False)):
        p = pos_x[i].astype(np.float32) / 255
        axes[0, k].imshow(p[H], cmap="gray", vmin=0, vmax=1); axes[0, k].set_title(f"centre slice (d={pm.box_d.iloc[i]:.1f} mm)", fontsize=8)
        axes[1, k].imshow(np.abs(p[H + 2] - p[H - 2]), cmap="magma", vmin=0, vmax=.5); axes[1, k].set_title("|z+2 − z−2|", fontsize=8)
        axes[0, k].axis("off"); axes[1, k].axis("off")
    plt.tight_layout(); plt.savefig(FIG / "exp000_25d_difference.png", dpi=130); plt.close()
    # 3. channel stats
    pc, nc, pmad, nmad = cs
    off = np.arange(C) - H
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(off, pc, "o-", color="#2563EB", label="positives"); ax[0].plot(off, nc, "s--", color="#64748B", label="random negatives")
    ax[0].set(title="Correlation with centre slice", xlabel="slice offset (mm)", ylim=(0.8, 1.01)); ax[0].legend()
    ax[1].plot(off, pmad, "o-", color="#2563EB"); ax[1].plot(off, nmad, "s--", color="#64748B")
    ax[1].set(title="Mean |difference| to centre slice", xlabel="slice offset (mm)")
    for (name, d), col in zip(prof.items(), ("#DC2626", "#D97706", "#0F766E")):
        v = np.array(d["mean_profile"]); ax[2].plot(off, v / v[H], "o-", color=col, label=f"{name} (n={d['n']})")
    ax[2].set(title="Nodule-core intensity along z (÷ centre)", xlabel="slice offset (mm)"); ax[2].legend(fontsize=7)
    for a in ax: a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(FIG / "exp000_25d_channel_stats.png", dpi=130); plt.close()


def main():
    cfg = load_config("configs/preprocessing.yaml")
    assert cfg["slice_context"]["input_slices"] == C and cfg["slice_context"]["slice_stride"] == 1
    scans = pd.read_csv(DATA / "metadata" / "scans.csv")
    cidx = pd.read_csv(DATA / "candidates" / "candidate_index.csv")
    ann = pd.read_csv(DATA / "annotations.csv")
    res = {"cache_info": json.loads((PROC / "patch_cache_info.json").read_text())}
    res["A_B_integrity_boundary"] = check_integrity_and_boundary(scans, cidx, cfg)
    res["B_all_boundary_candidates"] = check_all_boundary_candidates(scans, cidx, cfg)
    cmp_, pos_x, pm, neg_x, prof, cs = check_alignment_and_compare(scans, ann)
    res["C_D_alignment_and_comparison"] = cmp_
    figures(pos_x, pm, neg_x, prof, cs)
    Path("data/metadata/25d_validation.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "cache_info"}, indent=1))


if __name__ == "__main__":
    main()
