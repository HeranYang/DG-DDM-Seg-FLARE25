import os
import json
from glob import glob
from typing import Dict, List

import numpy as np
import nibabel as nib
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes, gaussian_filter, zoom, binary_erosion, distance_transform_edt
from nibabel.orientations import (
    axcodes2ornt,
    ornt_transform,
    apply_orientation,
)
import re

"""============================================================
本腳本在原版評估流程基礎上，新增 **Z 軸逆向處理**：
1. 先利用預處理紀錄，將 40‑slice 體積 **還原裁剪/補零**（crop ↔ pad）。
2. 再按 `scale_z` 將 Z 軸 **重採樣** 回與 GT 一致的 slice 數。
其餘流程（XY 還原、朝向恢復、指標計算）保持不變。
============================================================"""

# ------------------------------------------------------------------
# ★ 新增：Z 軸處理輔助函式
# ------------------------------------------------------------------

def undo_crop_pad_z(vol: np.ndarray, z_info: Dict) -> np.ndarray:
    """根據 `z_crop_pad` 字段逆向裁剪/補零，恢復到 `shape_after_z_resample`。
    * 若當初做的是 crop → 現在 pad 0 slice。
    * 若當初做的是 pad   → 現在 crop 掉該 0 slice。"""
    ct = z_info.get("crop_top", 0)
    cb = z_info.get("crop_bottom", 0)
    pt = z_info.get("pad_top", 0)
    pb = z_info.get("pad_bottom", 0)

    H, W, D = vol.shape
    # crop → pad 回來
    if ct or cb:
        new_D = D + ct + cb
        restored = np.zeros((H, W, new_D), dtype=vol.dtype)
        restored[:, :, ct : ct + D] = vol
        vol = restored
    # pad → crop 回去
    if pt or pb:
        vol = vol[:, :, pt : D - pb]
    return vol


def resample_z_axis(vol: np.ndarray, target_depth: int) -> np.ndarray:
    """最近鄰插值將 Z slice 數變為 `target_depth`"""
    cur_D = vol.shape[2]
    if cur_D == target_depth:
        return vol
    scale = target_depth / cur_D
    return zoom(vol, (1.0, 1.0, scale), order=0)

# ------------------------------------------------------------------
# ↓↓↓ 以下為原來腳本，僅在主流程中插入 Z 軸逆向處理 ↓↓↓
# ------------------------------------------------------------------

N_CLASS: int = 14                         # 背景=0，其餘 13 類
LABEL_SZ: int = 320                       # slice 預測 XY 分辨率

DO_KEEP_LARGEST_CC = False
DO_VOLUME_THRESHOLD = False
DO_FILL_HOLES = True
DO_GAUSSIAN_SMOOTH = False

MIN_VOL_VOX = [6000, 2500, 2500, 1000, 800, 600, 200, 200, 300, 200, 3500, 1500, 2500]
KEEP_NCC     = [1] * 13
GAUSS_SIGMA  = [1.2, 0.8, 1.0, 0.8, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 1.0, 0.8, 0.8]

# 路徑配置
LABEL_ROOT    = "../evaluation"                  # *.npy_fakeFusion.npy 所在目錄
GT_ROOT       = "../inputs/MRI_labelsVal"    # GT 標籤 NIfTI
SPACING_JSON  = "../data/spacing_record.json"   # 預處理紀錄
SAVE_DIR      = "../outputs"            # NIfTI 輸出
os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 原有工具函式（未改動，略過註釋）
# ------------------------------------------------------------------

def compute_dice(pred: np.ndarray, gt: np.ndarray, class_ids: list) -> dict:
    dice_scores = {}
    for cls in class_ids:
        pred_mask = (pred == cls)
        gt_mask   = (gt == cls)
        inter = np.logical_and(pred_mask, gt_mask).sum()
        size_sum = pred_mask.sum() + gt_mask.sum()
        dice = 1.0 if size_sum == 0 else 2 * inter / size_sum
        dice_scores[cls] = round(dice, 4)
    return dice_scores


def compute_nsd(pred: np.ndarray, gt: np.ndarray, spacing: list, class_ids: list, threshold_mm=1.0) -> dict:
    def get_surface(mask):
        eroded = binary_erosion(mask)
        return np.logical_and(mask, ~eroded)
    nsd_scores = {}
    for cls in class_ids:
        pred_mask = (pred == cls)
        gt_mask   = (gt == cls)
        if not pred_mask.any() and not gt_mask.any():
            nsd = 1.0
        elif not pred_mask.any() or not gt_mask.any():
            nsd = 0.0
        else:
            pred_surf = get_surface(pred_mask)
            gt_surf   = get_surface(gt_mask)
            dt_gt   = distance_transform_edt(~gt_surf, sampling=spacing)
            dt_pred = distance_transform_edt(~pred_surf, sampling=spacing)
            surf2gt = dt_gt[pred_surf]
            gt2surf = dt_pred[gt_surf]
            n_close = np.sum(surf2gt <= threshold_mm) + np.sum(gt2surf <= threshold_mm)
            n_total = surf2gt.size + gt2surf.size
            nsd = n_close / n_total if n_total else 0.0
        nsd_scores[cls] = round(nsd, 4)
    return nsd_scores


# ---------- 其他舊工具函式（無變更，僅貼關鍵邏輯） ----------

def invert_ornt(ornt_arr: np.ndarray) -> np.ndarray:
    inv = np.zeros_like(ornt_arr)
    for new_ax, (old_ax, flip) in enumerate(ornt_arr):
        inv[int(old_ax)] = [new_ax, flip]
    return inv

def sort_key(f):
    m = re.search(r"slice(\d+)", f)
    return int(m.group(1)) if m else -1

def load_spacing_dict(path: str) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def gather_slice_files(root: str) -> Dict[str, List[str]]:
    files = sorted(glob(os.path.join(root, "MR-id*-slice*.npy_fakeFusion.npy")))
    mapping: Dict[str, List[str]] = {}
    for f in files:
        sid = os.path.basename(f).split("-slice")[0]
        mapping.setdefault(sid, []).append(f)
    return mapping

def rebuild_volume(slices: List[str]) -> np.ndarray:
    num_slices = len(slices)
    vol_raw = np.zeros((LABEL_SZ, LABEL_SZ, num_slices, N_CLASS - 1), dtype=bool)
    for idx, sl_path in enumerate(sorted(slices, key=sort_key)):
        sl_pred = np.squeeze(np.load(sl_path))
        for cls in range(1, N_CLASS):
            vol_raw[:, :, idx, cls - 1] = sl_pred == cls
    vol_post = np.zeros_like(vol_raw)
    for cls in range(N_CLASS - 1):
        mask = vol_raw[..., cls]
        if DO_KEEP_LARGEST_CC:
            lbl = label(mask, connectivity=3)
            props = sorted(regionprops(lbl), key=lambda x: x.area, reverse=True)[: KEEP_NCC[cls]]
            tmp = np.zeros_like(mask)
            for p in props:
                tmp[lbl == p.label] = 1
            mask = tmp.astype(bool)
        if DO_VOLUME_THRESHOLD:
            lbl = label(mask, connectivity=3)
            tmp = np.zeros_like(mask)
            for rg in regionprops(lbl):
                if rg.area >= MIN_VOL_VOX[cls]:
                    tmp[lbl == rg.label] = 1
            mask = tmp.astype(bool)
        if DO_FILL_HOLES:
            mask = binary_fill_holes(mask)
        if DO_GAUSSIAN_SMOOTH:
            mask = gaussian_filter(mask.astype(float), sigma=GAUSS_SIGMA[cls]) > 0.5
        vol_post[..., cls] = mask
    lbl3d = np.argmax(vol_post, axis=3) + 1
    lbl3d[np.sum(vol_post, axis=3) == 0] = 0
    return lbl3d

# ---------- XY 逆向處理函式（保留舊版，略作兼容） ----------

def undo_pad_crop(label_vol: np.ndarray, spacing_info: Dict) -> np.ndarray:
    pad_info = spacing_info.get("label_pad") or spacing_info.get("pad_crop_xy", {}).get("lbl", {}).get("pad", {})
    crop_info = spacing_info.get("label_crop") or spacing_info.get("pad_crop_xy", {}).get("lbl", {}).get("crop", {})
    pad_info = {k: pad_info.get(k, 0) for k in ("top", "bottom", "left", "right")}
    crop_info = {k: crop_info.get(k, 0) for k in ("top", "left")}
    if crop_info["top"] or crop_info["left"]:
        ct, cl = crop_info["top"], crop_info["left"]
        H, W, D = label_vol.shape
        padded = np.zeros((H + ct * 2, W + cl * 2, D), dtype=label_vol.dtype)
        padded[ct : ct + H, cl : cl + W, :] = label_vol
        label_vol = padded
    if any(pad_info.values()):
        t, b, l, r = pad_info["top"], pad_info["bottom"], pad_info["left"], pad_info["right"]
        H, W, D = label_vol.shape
        label_vol = label_vol[t : H - b, l : W - r, :]
    return label_vol


def resize_to_original(label_vol: np.ndarray, spacing_info: Dict) -> np.ndarray:
    orig_H, orig_W, _ = spacing_info.get("original_shape", [*label_vol.shape[:2], 0])
    cur_H, cur_W, _ = label_vol.shape
    fy, fx = orig_H / cur_H, orig_W / cur_W
    resized = zoom(label_vol, (fy, fx, 1.0), order=0)
    dH, dW = resized.shape[0] - orig_H, resized.shape[1] - orig_W
    if dH or dW:
        tmp = np.zeros((orig_H, orig_W, resized.shape[2]), dtype=resized.dtype)
        h, w = min(orig_H, resized.shape[0]), min(orig_W, resized.shape[1])
        tmp[:h, :w, :] = resized[:h, :w, :]
        resized = tmp
    return resized


def restore_orientation(label_vol: np.ndarray, spacing_info: Dict) -> np.ndarray:
    if "ornt_transform" in spacing_info:
        inv_tr = invert_ornt(np.array(spacing_info["ornt_transform"], dtype=int))
        return apply_orientation(label_vol, inv_tr)
    orig_ax = tuple(spacing_info.get("orig_axcodes", [])) or tuple(spacing_info.get("original_axcodes", []))
    tgt_ax  = tuple(spacing_info.get("target_axcodes", []))
    if orig_ax and tgt_ax and orig_ax != tgt_ax:
        cur_ornt  = axcodes2ornt(tgt_ax)
        orig_ornt = axcodes2ornt(orig_ax)
        inv_tr = ornt_transform(cur_ornt, orig_ornt)
        return apply_orientation(label_vol, inv_tr)
    return label_vol

# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------

if __name__ == "__main__":
    spacing_dict = load_spacing_dict(SPACING_JSON)
    slice_map     = gather_slice_files(LABEL_ROOT)

    class_ids = list(range(1, N_CLASS))
    dice_sums   = {cls: 0.0 for cls in class_ids}
    dice_counts = {cls: 0   for cls in class_ids}

    for subject_id, slices in slice_map.items():
        print(subject_id)
        key = subject_id.replace("MR-id", "").zfill(4)
        info = spacing_dict.get(key)
        if info is None:
            print(f"⚠️  {subject_id}: 缺少 spacing 信息，已跳過。")
            continue

        # ---------- 1️⃣ slice → 3D vol（40 slices） ----------
        vol_pred = rebuild_volume(slices)
        vol_pred = np.transpose(vol_pred, (1, 0, 2))  # (H,W,D)

        # ---------- 2️⃣ Z 軸逆向處理 ----------
        z_info = info.get("z_crop_pad", info.get("z_crop", {}))
        vol_z_restored = undo_crop_pad_z(vol_pred, z_info)

        target_D = info.get("shape_after_xy", [None, None, None])[2]
        if target_D is None:
            # 兜底：依 scale_z 推算
            scale_z = info.get("scale_z", 1.0)
            target_D = int(round(vol_z_restored.shape[2] * (1 / scale_z)))
        vol_z_resampled = resample_z_axis(vol_z_restored, target_D)

        # ---------- 3️⃣ XY 還原 ----------
        vol_pc       = undo_pad_crop(vol_z_resampled, info)
        vol_resized  = resize_to_original(vol_pc, info)
        vol_restored = restore_orientation(vol_resized, info)

        # ---------- 4️⃣ 讀 GT & 指標 ----------
        gt_path = os.path.join(GT_ROOT, info.get("label_file", ""))
        if not os.path.isfile(gt_path):
            print(f"⚠️  找不到 GT: {gt_path}")
            continue
        gt_nii   = nib.load(gt_path)
        gt_data  = gt_nii.get_fdata()
        affine   = gt_nii.affine
        dtype    = gt_nii.get_data_dtype()

        dice_res = compute_dice(vol_restored, gt_data, class_ids)
        print(f"📊 {subject_id} DICE: {dice_res}")
        for cls in class_ids:
            dice_sums[cls]   += dice_res.get(cls, 0.0)
            dice_counts[cls] += 1

        # ---------- 5️⃣ 保存 NIfTI ----------
        orig_fn      = info.get("label_file", subject_id) .rsplit(".nii", 1)[0]
        save_path    = os.path.join(SAVE_DIR, f"{orig_fn}_restored.nii.gz")
        nib.save(nib.Nifti1Image(vol_resized.astype(dtype), affine), save_path)
        print(f"✅ {subject_id}: 輸出完成 → {save_path}\n")

    # ---------- 6️⃣ 統計平均 DICE ----------
    print("\n🎯 每類 DICE 平均值：")
    valid = []
    for cls in class_ids:
        if dice_counts[cls]:
            avg = dice_sums[cls] / dice_counts[cls]
            valid.append(avg)
            print(f"  ➤ 類別 {cls}: {avg:.4f}")
        else:
            print(f"  ⚠️ 類別 {cls} 無有效預測，跳過。")
    if valid:
        print(f"\n📊 Macro Avg DICE: {np.mean(valid):.4f}")
    else:
        print("\n⚠️ 沒有任何有效預測，無法計算 Macro Avg DICE。")



