#!/usr/bin/env python3
"""
Renca Transwell Migration — Batch Cell Counter + Normalization
Adapted from the PRC CELLCOUNTER pipeline (watershed + intensity correction).

Folders: c1,c2,c3 (Control) / s1,s2,s3 (Survivin)
For s2 and s3, uses "second time taking photos" subfolder if present.
"""

import os, sys, glob, csv, numpy as np

try:
    import cv2
except ImportError:
    os.system(f"{sys.executable} -m pip install opencv-python-headless")
    import cv2

try:
    from skimage import morphology, measure, segmentation
    from skimage.feature import peak_local_max
except ImportError:
    os.system(f"{sys.executable} -m pip install scikit-image")
    from skimage import morphology, measure, segmentation
    from skimage.feature import peak_local_max

try:
    from scipy import ndimage
except ImportError:
    os.system(f"{sys.executable} -m pip install scipy")
    from scipy import ndimage

try:
    import tifffile
except ImportError:
    os.system(f"{sys.executable} -m pip install tifffile")
    import tifffile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE_DIR, "cellcounter_results")
os.makedirs(OUT_DIR, exist_ok=True)

PAPER_W, PAPER_H = 1280, 960
CELL_RADIUS_R = 8
NOISE_THRESHOLD = 4

# Experiment pairings: (ct_folder, surv_folder, experiment_name)
EXPERIMENTS = [
    ("c1", "s1", "Exp1"),
    ("c2", "s2", "Exp2"),
    ("c3", "s3", "Exp3"),
]

# ═══════════════════════════════════════════════════════════════
#  PIPELINE FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def maximum_entropy_threshold(gray):
    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    hist = hist / hist.sum()
    best_thresh, best_entropy = 0, -np.inf
    for t in range(1, 255):
        p_bg = hist[:t]; s_bg = p_bg.sum()
        if s_bg == 0 or s_bg == 1: continue
        p_bg_n = p_bg / s_bg; p_bg_n = p_bg_n[p_bg_n > 0]
        h_bg = -np.sum(p_bg_n * np.log(p_bg_n))
        p_fg = hist[t:]; s_fg = p_fg.sum()
        if s_fg == 0 or s_fg == 1: continue
        p_fg_n = p_fg / s_fg; p_fg_n = p_fg_n[p_fg_n > 0]
        h_fg = -np.sum(p_fg_n * np.log(p_fg_n))
        total = h_bg + h_fg
        if total > best_entropy:
            best_entropy = total; best_thresh = t
    return best_thresh


def watershed_intensity_count(cluster_mask, gray_img, bg_intensity,
                              median_single_area, median_single_stain,
                              cell_radius):
    dist = ndimage.distance_transform_edt(cluster_mask)
    coords = peak_local_max(dist,
                            min_distance=max(4, cell_radius // 2),
                            threshold_abs=max(dist.max() * 0.2, 2),
                            labels=cluster_mask)
    if len(coords) < 2:
        area = float(cluster_mask.sum())
        stain = bg_intensity - float(np.mean(gray_img[cluster_mask > 0]))
        area_ratio = area / median_single_area
        int_ratio = max(1.0, stain / median_single_stain)
        return max(1, int(round(area_ratio * int_ratio))), 'intensity_only'

    markers = np.zeros_like(cluster_mask, dtype=np.int32)
    for idx, (y, x) in enumerate(coords, 1):
        markers[y, x] = idx
    markers = ndimage.label(
        morphology.dilation(markers > 0, morphology.disk(2))
    )[0]
    ws_labels = segmentation.watershed(-dist, markers, mask=cluster_mask)

    total = 0
    sub_props = measure.regionprops(ws_labels, intensity_image=gray_img)
    for sub in sub_props:
        sub_stain = bg_intensity - sub.mean_intensity
        area_r = sub.area / median_single_area
        int_r = max(1.0, sub_stain / median_single_stain)
        if area_r < 0.3:
            continue
        elif area_r <= 1.5:
            sub_count = max(1, int(round(int_r)))
        else:
            sub_count = max(1, int(round(area_r * int_r)))
        total += sub_count
    return max(1, total), f'watershed({len(sub_props)}sub)'


def save_overlay(rgb, labels, count, path, title=""):
    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax.imshow(rgb)
    for region in measure.regionprops(labels):
        mask_r = (labels == region.label).astype(np.uint8)
        contours, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            cs = c.squeeze()
            if cs.ndim == 2 and len(cs) > 2:
                ax.plot(cs[:, 0], cs[:, 1], color='lime', linewidth=1.2, alpha=0.9)
                ax.plot([cs[-1, 0], cs[0, 0]], [cs[-1, 1], cs[0, 1]],
                        color='lime', linewidth=1.2, alpha=0.9)
        cy, cx = region.centroid
        ax.text(cx, cy, str(region.label), color='yellow', fontsize=5,
                fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.6, edgecolor='none'))
    ax.set_title(f"{title}\nTotal Cells: {count}", fontsize=16, fontweight='bold', color='white')
    ax.axis('off')
    fig.patch.set_facecolor('#1a1a2e')
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close(fig)


def process_image(img_path, name, out_dir):
    print(f"  Processing: {name}")
    if img_path.lower().endswith(('.tif', '.tiff')):
        raw = tifffile.imread(img_path)
    else:
        raw = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    orig_h, orig_w = raw.shape[:2]
    if len(raw.shape) == 2:
        rgb = np.stack([raw]*3, axis=-1)
        gray_orig = raw
    else:
        rgb = raw[:,:,:3] if raw.shape[2] >= 3 else np.stack([raw[:,:,0]]*3, axis=-1)
        gray_orig = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    gray = cv2.resize(gray_orig, (PAPER_W, PAPER_H), interpolation=cv2.INTER_AREA)
    rgb_r = cv2.resize(rgb.astype(np.uint8), (PAPER_W, PAPER_H), interpolation=cv2.INTER_AREA)

    # Stage 1: Maximum Entropy
    gray_s = cv2.GaussianBlur(gray, (5, 5), 1.5)
    thresh_val = maximum_entropy_threshold(gray_s)
    _, binary_s1 = cv2.threshold(gray_s, thresh_val, 255, cv2.THRESH_BINARY_INV)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    s1c = cv2.morphologyEx(binary_s1, cv2.MORPH_OPEN, k3, iterations=1)
    s1c = cv2.morphologyEx(s1c, cv2.MORPH_CLOSE, k3, iterations=1)
    s1b = morphology.remove_small_objects(s1c.astype(bool), min_size=30)
    s1c = s1b.astype(np.uint8) * 255

    # Stage 2: Otsu pore exclusion
    cell_px = gray_s[s1c > 0]
    if len(cell_px) > 0:
        otsu_val, _ = cv2.threshold(cell_px, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        s2 = np.zeros_like(gray_s, dtype=np.uint8)
        masked_gray = gray_s.copy(); masked_gray[s1c == 0] = 255
        s2[masked_gray <= otsu_val] = 255
        s2 = cv2.bitwise_and(s2, s1c)
    else:
        s2 = s1c.copy()

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    s2 = cv2.morphologyEx(s2, cv2.MORPH_OPEN, k3, iterations=1)
    s2 = cv2.morphologyEx(s2, cv2.MORPH_CLOSE, k5, iterations=1)
    s2b = morphology.remove_small_objects(s2.astype(bool), min_size=50)
    s2b = morphology.remove_small_holes(s2b, area_threshold=200)
    s2 = s2b.astype(np.uint8) * 255

    # Stage 3: Watershed + intensity counting
    labeled = measure.label(s2 > 0)
    props = measure.regionprops(labeled, intensity_image=gray_s)
    bg_intensity = float(np.mean(gray_s[s2 == 0]))

    singles, cluster_list = [], []
    for region in props:
        cmask = (labeled == region.label).astype(np.uint8)
        mic_r = ndimage.distance_transform_edt(cmask).max()
        stain_depth = bg_intensity - region.mean_intensity
        if mic_r < NOISE_THRESHOLD:
            continue
        elif mic_r <= CELL_RADIUS_R:
            singles.append((region, mic_r, stain_depth))
        else:
            cluster_list.append((region, mic_r, stain_depth))

    if singles:
        median_single_area = float(np.median([r.area for r, _, _ in singles]))
        median_single_stain = float(np.median([s for _, _, s in singles]))
    else:
        median_single_area = 200.0
        median_single_stain = 30.0
    median_single_stain = max(median_single_stain, 1.0)

    total = 0
    final_lbl = np.zeros_like(labeled, dtype=np.int32)

    for region, mic_r, stain in singles:
        total += 1
        final_lbl[labeled == region.label] = total

    for region, mic_r, stain in cluster_list:
        cluster_mask = (labeled == region.label).astype(np.uint8)
        cc, method = watershed_intensity_count(
            cluster_mask, gray_s, bg_intensity,
            median_single_area, median_single_stain, CELL_RADIUS_R)
        total += cc
        final_lbl[labeled == region.label] = total

    print(f"    → {total} cells ({len(singles)} singles, {len(cluster_list)} clusters)")

    save_overlay(rgb_r, final_lbl, total,
                 os.path.join(out_dir, f"{name}_overlay.png"), title=name)
    return total


def get_image_folder(folder_name):
    """Get the actual image folder, checking for 'second time' subfolder."""
    folder_path = os.path.join(BASE_DIR, folder_name)
    # Check for "second time taking photos" subfolder
    for sub in ["second time taking photos", "Second time taking photos"]:
        sub_path = os.path.join(folder_path, sub)
        if os.path.isdir(sub_path):
            # Check if subfolder has images
            imgs = glob.glob(os.path.join(sub_path, "*.tif"))
            if imgs:
                print(f"  Using subfolder: {sub}")
                return sub_path
    return folder_path


def process_folder(folder_name):
    """Process all images in a folder. Returns list of cell counts."""
    folder_path = get_image_folder(folder_name)
    folder_out = os.path.join(OUT_DIR, folder_name)
    os.makedirs(folder_out, exist_ok=True)

    all_files = sorted(
        glob.glob(os.path.join(folder_path, "*.tif")) +
        glob.glob(os.path.join(folder_path, "*.tiff")) +
        glob.glob(os.path.join(folder_path, "*.jpg"))
    )
    # Skip macOS resource fork files
    all_files = [f for f in all_files if not os.path.basename(f).startswith("._")]

    if not all_files:
        print(f"  WARNING: No images in {folder_path}")
        return []

    print(f"  Found {len(all_files)} images")
    counts = []
    for fpath in all_files:
        fname = os.path.splitext(os.path.basename(fpath))[0]
        safe = fname.replace(" ", "_").replace("(", "").replace(")", "")
        count = process_image(fpath, safe, folder_out)
        counts.append(count)
    return counts


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*70)
    print("Renca Transwell Migration — Batch Cell Counter")
    print("="*70)

    all_results = {}

    for ct_folder, surv_folder, exp_name in EXPERIMENTS:
        for folder in [ct_folder, surv_folder]:
            print(f"\n{'='*60}")
            print(f"[{exp_name}] Folder: {folder}")
            print(f"{'='*60}")
            counts = process_folder(folder)
            all_results[folder] = counts
            if counts:
                print(f"  Counts: {counts}")
                print(f"  Mean: {np.mean(counts):.1f} ± {np.std(counts):.1f}")

    # ── Raw counts summary ──
    print(f"\n\n{'='*70}")
    print("RAW CELL COUNTS")
    print(f"{'='*70}")
    print(f"{'Folder':<10} {'Counts':<40} {'Mean':>8} {'SD':>8}")
    print("-"*70)
    for ct_f, surv_f, _ in EXPERIMENTS:
        for folder in [ct_f, surv_f]:
            counts = all_results.get(folder, [])
            if counts:
                vals = ", ".join(str(c) for c in counts)
                print(f"{folder:<10} {vals:<40} {np.mean(counts):>8.1f} {np.std(counts):>8.1f}")

    # ── Normalize ──
    print(f"\n{'='*70}")
    print("NORMALIZATION (each value / CT mean of same experiment)")
    print(f"{'='*70}")

    normalized = {}
    for ct_folder, surv_folder, exp_name in EXPERIMENTS:
        ct_counts = all_results.get(ct_folder, [])
        surv_counts = all_results.get(surv_folder, [])
        if not ct_counts:
            continue
        ct_mean = np.mean(ct_counts)
        print(f"\n[{exp_name}] CT mean = {ct_mean:.1f}")
        ct_norm = [c / ct_mean for c in ct_counts]
        normalized[ct_folder] = ct_norm
        surv_norm = [c / ct_mean for c in surv_counts]
        normalized[surv_folder] = surv_norm
        print(f"  {ct_folder:<8} norm={[f'{v:.4f}' for v in ct_norm]}")
        print(f"  {surv_folder:<8} norm={[f'{v:.4f}' for v in surv_norm]}")

    # ── Pool & export ──
    all_ct_norm = []
    all_surv_norm = []
    for ct_f, surv_f, _ in EXPERIMENTS:
        all_ct_norm.extend(normalized.get(ct_f, []))
        all_surv_norm.extend(normalized.get(surv_f, []))

    print(f"\n{'='*70}")
    print("COMBINED NORMALIZED DATA")
    print(f"{'='*70}")
    print(f"CT   (n={len(all_ct_norm)}):  {[f'{v:.4f}' for v in all_ct_norm]}")
    print(f"Surv (n={len(all_surv_norm)}): {[f'{v:.4f}' for v in all_surv_norm]}")
    if all_ct_norm:
        print(f"CT   Mean ± SD: {np.mean(all_ct_norm):.4f} ± {np.std(all_ct_norm):.4f}")
    if all_surv_norm:
        print(f"Surv Mean ± SD: {np.mean(all_surv_norm):.4f} ± {np.std(all_surv_norm):.4f}")

    # Prism CSV
    prism_path = os.path.join(BASE_DIR, "renca_transwell_normalized_for_prism.csv")
    max_n = max(len(all_ct_norm), len(all_surv_norm))
    with open(prism_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["CT (Normalized)", "Survivin (Normalized)"])
        for i in range(max_n):
            ct_val = f"{all_ct_norm[i]:.4f}" if i < len(all_ct_norm) else ""
            surv_val = f"{all_surv_norm[i]:.4f}" if i < len(all_surv_norm) else ""
            w.writerow([ct_val, surv_val])
    print(f"\n✓ Prism CSV: {prism_path}")

    # Raw counts CSV
    raw_path = os.path.join(BASE_DIR, "renca_transwell_raw_counts.csv")
    with open(raw_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Folder", "Image", "Raw_Count"])
        for ct_f, surv_f, exp_name in EXPERIMENTS:
            for folder in [ct_f, surv_f]:
                for i, c in enumerate(all_results.get(folder, []), 1):
                    w.writerow([folder, f"Image_{i}", c])
    print(f"✓ Raw CSV: {raw_path}")

    # Detailed CSV
    detail_path = os.path.join(BASE_DIR, "renca_transwell_detailed_results.csv")
    with open(detail_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Experiment", "Condition", "Image", "Raw_Count", "CT_Mean", "Normalized"])
        for ct_f, surv_f, exp_name in EXPERIMENTS:
            ct_counts = all_results.get(ct_f, [])
            surv_counts = all_results.get(surv_f, [])
            if not ct_counts: continue
            ct_mean = np.mean(ct_counts)
            for i, c in enumerate(ct_counts, 1):
                w.writerow([exp_name, "CT", f"Image_{i}", c, f"{ct_mean:.1f}", f"{c/ct_mean:.4f}"])
            for i, c in enumerate(surv_counts, 1):
                w.writerow([exp_name, "Survivin", f"Image_{i}", c, f"{ct_mean:.1f}", f"{c/ct_mean:.4f}"])
    print(f"✓ Detailed CSV: {detail_path}")

    print(f"\n{'='*70}")
    print("ALL DONE!")
    print(f"{'='*70}")
