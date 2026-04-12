#!/usr/bin/env python3
"""Run SRI24 atlas-based lobar feature extraction for the UCSF cohort."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt
import numpy as np

from extract_sri24_atlas_features import (
    compute_feature_row,
    load_3d_volume,
    load_atlas_context,
    project_atlas_to_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--structural-dir",
        type=Path,
        default=Path("/media/pleb/7AFAFDB8FAFD70AD/UPenn/UCSF/DATA-IMAGE-STRUCTURAL"),
        help="Directory containing one folder per UCSF patient.",
    )
    parser.add_argument(
        "--segmentation-dir",
        type=Path,
        default=Path("/media/pleb/7AFAFDB8FAFD70AD/UPenn/UCSF/DATA-AUTOMATED-SEGMENT"),
        help="Directory containing UCSF tumor segmentation files.",
    )
    parser.add_argument(
        "--atlas-dir",
        type=Path,
        default=Path(__file__).parent / "sri24_atlas",
        help="Directory containing the SRI24 atlas files (tzo116plus.nii.gz, suptent.nii.gz, etc.).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("/home/pleb/Codes/GBM-multimodal-clustering/UCSF_sri24_atlas_features_all.csv"),
        help="Path for the cohort-level CSV output.",
    )
    parser.add_argument(
        "--dilation-voxels",
        type=int,
        default=3,
        help="Binary dilation applied to the supratentorial atlas mask before lobe fill.",
    )
    parser.add_argument(
        "--viz-dir",
        type=Path,
        default=None,
        help=(
            "If supplied, one PNG per patient is saved here showing the atlas "
            "overlay on the segmentation (axial / sagittal / coronal). "
            "Example: outputs/atlas_overlays"
        ),
    )
    return parser.parse_args()


def preferred_segmentation_path(segmentation_dir: Path, case_id: str) -> Path | None:
    candidates = [
        segmentation_dir / f"{case_id}_tumor_segmentation.nii",
        segmentation_dir / f"{case_id}_tumor_segmentation.nii.gz",
        segmentation_dir / f"{case_id}_automated_approx_segm.nii",
        segmentation_dir / f"{case_id}_automated_approx_segm.nii.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def write_rows(rows: list[dict[str, object]], out_csv: Path) -> None:
    if not rows:
        raise ValueError("No feature rows were produced.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

# Segmentation label → opaque RGBA color [0..1]
# Labels: 1=Necrotic Core, 2=Edema, 4=Enhancing Tumor
_SEG_COLORS: dict[int, tuple[float, float, float, float]] = {
    1: (0.00, 0.90, 0.90, 1.0),  # NC  — cyan
    2: (1.00, 0.85, 0.00, 1.0),  # ED  — amber
    4: (1.00, 0.25, 0.25, 1.0),  # EN  — red
}

# Atlas lobe id → semitransparent RGBA overlay color
# IDs: 1=frontal, 2=temporal, 3=parietal, 4=occipital
_ATLAS_COLORS: dict[int, tuple[float, float, float, float]] = {
    1: (0.20, 0.60, 1.00, 0.40),  # frontal   — steel blue
    2: (0.20, 0.85, 0.45, 0.40),  # temporal  — mint
    3: (1.00, 0.60, 0.10, 0.40),  # parietal  — orange
    4: (0.90, 0.25, 0.90, 0.40),  # occipital — violet
}


def _best_slice(volume: np.ndarray, axis: int) -> int:
    """Return the index along *axis* whose 2-D cross-section has the most
    non-zero voxels.  Falls back to the midpoint when the volume is empty.

    Why atlas-coverage over middle slice
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The brain occupies only part of the bounding box stored in a NIfTI file,
    and a tumour can sit far off-centre.  Choosing the slice with the greatest
    number of labelled atlas voxels guarantees we land in a location where the
    atlas projection has actually assigned labels — the very thing we want to
    inspect.  A fixed middle slice frequently falls in unlabelled white-matter
    or outside the tumour ROI entirely.
    """
    counts = np.count_nonzero(
        volume, axis=tuple(i for i in range(volume.ndim) if i != axis)
    )
    if counts.max() == 0:
        return volume.shape[axis] // 2
    return int(np.argmax(counts))


def _make_seg_rgba(seg_slice: np.ndarray) -> np.ndarray:
    """Build an H×W×4 float32 RGBA image from a segmentation label slice.

    Each known tumor label gets its own bright, fully-opaque color.
    Background (label 0) is left transparent so the black axes show through.
    This guarantees tumor regions are always visible regardless of label value.
    """
    h, w = seg_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    for label, color in _SEG_COLORS.items():
        rgba[seg_slice == label] = color
    return rgba


def _composite_atlas(base_rgba: np.ndarray, atlas_slice: np.ndarray) -> np.ndarray:
    """Porter-Duff 'over' composite of atlas lobe tints onto *base_rgba*.

    WHY THE PREVIOUS APPROACH ONLY SHOWED THE ATLAS
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The old code rendered tumor_data (labels 1/2/4) with cmap='gray' and no
    vmin/vmax.  Matplotlib maps value 0→black and max→white; with labels
    only up to 4 (in a uint8 volume), every tumor voxel rendered as nearly
    pure black — indistinguishable from the black axes background.  The atlas
    overlay at alpha=0.4 then painted a solid tint over an invisible base,
    making it appear as standalone atlas colors with nothing underneath.

    HOW THIS FIXES IT
    ~~~~~~~~~~~~~~~~~
    1. Segmentation is rendered as bright, per-label RGBA colors (always
       visible, label-value-independent).
    2. Atlas overlay is blended via Porter-Duff 'over':
         out_A = src_A + dst_A*(1-src_A)
         out_C = (src_C*src_A + dst_C*dst_A*(1-src_A)) / out_A
       This mathematically guarantees ≥60% of the tumor color shows through
       wherever both tumor and atlas regions overlap.
    """
    out = base_rgba.copy()
    for lobe_id, (r, g, b, a_src) in _ATLAS_COLORS.items():
        mask = atlas_slice == lobe_id
        if not mask.any():
            continue
        a_dst = out[mask, 3]
        a_out = a_src + a_dst * (1.0 - a_src)
        np.clip(a_out, 0.0, 1.0, out=a_out)
        safe_denom = np.where(a_out > 0, a_out, 1.0)
        for ch, val in enumerate((r, g, b)):
            c_dst = out[mask, ch]
            out[mask, ch] = np.where(
                a_out > 0,
                (val * a_src + c_dst * a_dst * (1.0 - a_src)) / safe_denom,
                0.0,
            )
        out[mask, 3] = a_out
    return out


def generate_overlay_visualization(
    tumor_data: np.ndarray,
    atlas_labels: np.ndarray,
    case_id: str,
    output_dir: Path,
) -> None:
    """Save a PNG with axial / sagittal / coronal atlas-overlay views.

    Parameters
    ----------
    tumor_data:
        3-D uint8 segmentation array (H × W × D).
        Expected labels: 0=background, 1=NC, 2=ED, 4=EN.
    atlas_labels:
        3-D uint8 array of the same shape — lobar atlas labels projected
        into segmentation space (0=bg, 1=frontal, 2=temporal, 3=parietal,
        4=occipital).
    case_id:
        Patient identifier used as the filename stem and figure title.
    output_dir:
        Directory in which the PNG is written (created if absent).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------- #
    # Slice selection: pick the cross-section with most atlas coverage per axis
    # axis 2 → axial (Z)  |  axis 0 → sagittal (X)  |  axis 1 → coronal (Y)
    # ---------------------------------------------------------------------- #
    ax_z = _best_slice(atlas_labels, axis=2)
    ax_x = _best_slice(atlas_labels, axis=0)
    ax_y = _best_slice(atlas_labels, axis=1)

    views = [
        (tumor_data[:, :, ax_z], atlas_labels[:, :, ax_z], f"Axial  (z={ax_z})"),
        (tumor_data[ax_x, :, :], atlas_labels[ax_x, :, :], f"Sagittal (x={ax_x})"),
        (tumor_data[:, ax_y, :], atlas_labels[:, ax_y, :], f"Coronal (y={ax_y})"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0d0d0d")
    fig.suptitle(case_id, color="white", fontsize=13, fontweight="bold", y=1.02)

    for ax, (seg_slice, atl_slice, title) in zip(axes, views):
        ax.set_facecolor("black")
        ax.set_title(title, color="#cccccc", fontsize=9, pad=4)
        ax.axis("off")

        # Transpose: aligns NIfTI storage axes to radiological display
        # convention (inferior at bottom, left on left).
        seg_t = seg_slice.T
        atl_t = atl_slice.T

        # 1. Build per-label opaque RGBA for tumor segmentation
        base_rgba = _make_seg_rgba(seg_t)

        # 2. Porter-Duff 'over': blend atlas lobe tints onto the tumor colors
        composite = _composite_atlas(base_rgba, atl_t)

        # 3. Entirely empty panel — show a faint indicator
        if composite[:, :, 3].max() == 0:
            ax.text(
                0.5, 0.5, "no data",
                color="gray", fontsize=8,
                ha="center", va="center",
                transform=ax.transAxes,
            )
            continue

        ax.imshow(composite, origin="lower", interpolation="nearest")

        # 4. Contour outlines per tumor sub-region for extra clarity
        for label in (1, 2, 4):
            mask = (seg_t == label).astype(np.float32)
            if mask.any():
                r, g, b, _ = _SEG_COLORS[label]
                ax.contour(
                    mask, levels=[0.5],
                    colors=[(r, g, b)],
                    linewidths=0.6,
                    origin="lower",
                )

    # Legend at bottom
    legend_entries = [
        ("NC",       _SEG_COLORS[1][:3]),
        ("ED",       _SEG_COLORS[2][:3]),
        ("EN",       _SEG_COLORS[4][:3]),
        ("Frontal",  _ATLAS_COLORS[1][:3]),
        ("Temporal", _ATLAS_COLORS[2][:3]),
        ("Parietal", _ATLAS_COLORS[3][:3]),
        ("Occipital",_ATLAS_COLORS[4][:3]),
    ]
    patches = [
        plt.Rectangle((0, 0), 1, 1, color=rgb, label=name)
        for name, rgb in legend_entries
    ]
    fig.legend(
        handles=patches,
        loc="lower center",
        ncol=len(patches),
        fontsize=7,
        framealpha=0.2,
        facecolor="#1a1a1a",
        labelcolor="white",
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=0.8,
        bbox_to_anchor=(0.5, -0.04),
    )

    plt.tight_layout(pad=0.5)
    out_path = output_dir / f"{case_id}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    args = parse_args()

    rows: list[dict[str, object]] = []
    missing_segmentations: list[str] = []
    case_dirs = sorted(path for path in args.structural_dir.iterdir() if path.is_dir())
    atlas = load_atlas_context(args.atlas_dir, args.dilation_voxels)
    projected_cache = None
    projected_shape = None
    projected_affine = None
    reused_projection_count = 0
    rebuilt_projection_count = 0

    for index, case_dir in enumerate(case_dirs, start=1):
        case_id = case_dir.name
        tumor_seg_path = preferred_segmentation_path(args.segmentation_dir, case_id)
        if tumor_seg_path is None:
            missing_segmentations.append(case_id)
            continue

        tumor_img, tumor_data = load_3d_volume(tumor_seg_path)
        current_shape = tuple(tumor_img.shape[:3])
        current_affine = tumor_img.affine

        if (
            projected_cache is None
            or projected_shape != current_shape
            or not np.allclose(projected_affine, current_affine, atol=1e-4)
        ):
            projected_cache = project_atlas_to_target(atlas, tumor_img)
            projected_shape = current_shape
            projected_affine = current_affine.copy()
            rebuilt_projection_count += 1
        else:
            reused_projection_count += 1

        row = compute_feature_row(
            tumor_data=tumor_data.astype(np.uint8),
            tumor_seg_path=tumor_seg_path,
            case_id=case_id,
            projected=projected_cache,
        )
        row["patient_id"] = case_id
        row["structural_folder"] = str(case_dir)
        row["segmentation_file_used"] = str(tumor_seg_path)
        rows.append(row)

        # ---- optional atlas-overlay visualization ----------------------
        if args.viz_dir is not None:
            try:
                generate_overlay_visualization(
                    tumor_data=tumor_data,
                    atlas_labels=projected_cache.lobar_labels,
                    case_id=case_id,
                    output_dir=args.viz_dir,
                )
            except Exception as exc:  # noqa: BLE001
                # Visualization failure must never abort feature extraction
                print(f"  [viz] WARNING: could not generate overlay for {case_id}: {exc}", flush=True)

        if index % 25 == 0 or index == len(case_dirs):
            print(f"Processed {index}/{len(case_dirs)} patients", flush=True)

    if missing_segmentations:
        print(f"Skipped {len(missing_segmentations)} patients without segmentations", flush=True)

    print(
        f"Atlas projections rebuilt={rebuilt_projection_count}, reused={reused_projection_count}",
        flush=True,
    )
    write_rows(rows, args.out_csv)
    print(f"Wrote cohort features to {args.out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
