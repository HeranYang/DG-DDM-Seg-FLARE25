import os
import json
from glob import glob
from typing import Tuple, Dict

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from nibabel.orientations import (
    aff2axcodes,
    axcodes2ornt,
    ornt_transform,
    apply_orientation,
)

MRI_IMG_DIR = "./inputs/MRI_imagesVal"
MRI_LBL_DIR = "./inputs/MRI_labelsVal"
MRI_OUT_DIR = "./data/abdominal/CHAOST2/processed"
SPACING_JSON_PATH = "./data/spacing_record.json"

TARGET_SPACING_XY = 1.0   
TARGET_SPACING_Z = 5.0  
INTERMEDIATE_SHAPE = 320 
FINAL_SHAPE = 320        
TARGET_SLICES_Z = 48     

os.makedirs(MRI_OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(SPACING_JSON_PATH), exist_ok=True)



def clip_and_normalize_intensity(vol: np.ndarray, clip_percentile: float = 99.5) -> np.ndarray:
    upper = np.percentile(vol, clip_percentile)
    vol = np.clip(vol, 0, upper)
    mean, std = vol.mean(), vol.std()
    return ((vol - mean) / std).astype(np.float32)


def resample_xy_slices(vol: np.ndarray, scale_x: float, scale_y: float, mode: str) -> np.ndarray:
    H, W, D = vol.shape
    new_h, new_w = int(round(H * scale_y)), int(round(W * scale_x))
    resampled = [
        F.interpolate(
            torch.from_numpy(vol[:, :, i]).unsqueeze(0).unsqueeze(0).float(),
            size=(new_h, new_w),
            mode=mode,
            align_corners=False if mode == "bilinear" else None,
        ).squeeze().numpy()
        for i in range(D)
    ]
    return np.stack(resampled, axis=2)


def resample_z(vol: np.ndarray, scale_z: float, mode: str) -> np.ndarray:
    if np.isclose(scale_z, 1.0):
        return vol  # 不需重採樣

    # vol: H×W×D → D×H×W
    vol_t = torch.from_numpy(vol).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).float()
    new_d = int(round(vol_t.shape[2] * scale_z))

    vol_rs = F.interpolate(
        vol_t,
        size=(new_d, vol_t.shape[3], vol_t.shape[4]),
        mode="trilinear" if mode == "bilinear" else "nearest",
        align_corners=False if mode == "bilinear" else None,
    )

    # 回到 H×W×D'
    return vol_rs.squeeze().permute(1, 2, 0).numpy()


def pad_or_crop_xy(vol: np.ndarray, target_size: int) -> Tuple[np.ndarray, Dict]:
    H, W, D = vol.shape
    record: Dict[str, Dict] = {}

    # padding
    pad_h, pad_w = max(0, target_size - H), max(0, target_size - W)
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    record["pad"] = {"top": pad_top, "bottom": pad_bottom, "left": pad_left, "right": pad_right}
    vol = np.pad(vol, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant")

    # cropping
    H, W = vol.shape[:2]
    crop_top, crop_left = max(0, (H - target_size) // 2), max(0, (W - target_size) // 2)
    record["crop"] = {"top": crop_top, "left": crop_left}
    vol = vol[crop_top: crop_top + target_size, crop_left: crop_left + target_size, :]
    return vol, record


def crop_or_pad_z(img_vol: np.ndarray, lbl_vol: np.ndarray, target_d: int = 40) -> Tuple[np.ndarray, np.ndarray, Dict]:
    assert img_vol.shape == lbl_vol.shape
    H, W, D = img_vol.shape

    # 初始化記錄
    crop_top = crop_bottom = pad_top = pad_bottom = 0

    if D > target_d:
        # 從中央裁剪
        crop_top = (D - target_d) // 2
        crop_bottom = D - target_d - crop_top
        img_vol = img_vol[:, :, crop_top: crop_top + target_d]
        lbl_vol = lbl_vol[:, :, crop_top: crop_top + target_d]
    elif D < target_d:
        # 兩端補零
        pad_top = (target_d - D) // 2
        pad_bottom = target_d - D - pad_top
        img_vol = np.pad(img_vol, ((0, 0), (0, 0), (pad_top, pad_bottom)), mode="constant")
        lbl_vol = np.pad(lbl_vol, ((0, 0), (0, 0), (pad_top, pad_bottom)), mode="constant")

    record = {
        "crop_top": crop_top,
        "crop_bottom": crop_bottom,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
    }
    return img_vol, lbl_vol, record


def get_new_affine(orig_aff: np.ndarray,
                   spacing_xy: float,
                   spacing_z: float,
                   z_shift: int = 0) -> np.ndarray:
    new_aff = orig_aff.copy()

    # X/Y spacing
    for i in range(2):
        direction = orig_aff[:3, i] / np.linalg.norm(orig_aff[:3, i])
        new_aff[:3, i] = direction * spacing_xy

    # Z spacing
    direction_z = orig_aff[:3, 2] / np.linalg.norm(orig_aff[:3, 2])
    new_aff[:3, 2] = direction_z * spacing_z

    # 原點因裁剪上移 z_shift * spacing_z
    if z_shift > 0:
        new_aff[:3, 3] += direction_z * spacing_z * z_shift

    return new_aff


def preprocess_xy(vol: np.ndarray,
                  spacing_x: float,
                  spacing_y: float,
                  mode: str,
                  do_norm: bool) -> Tuple[np.ndarray, Dict]:
    record: Dict[str, object] = {}

    if do_norm:
        vol = clip_and_normalize_intensity(vol)

    # 初次重採樣 X/Y
    scale_x, scale_y = spacing_x / TARGET_SPACING_XY, spacing_y / TARGET_SPACING_XY
    record["scale_xy"] = {"x": scale_x, "y": scale_y}
    vol_xy = resample_xy_slices(vol, scale_x, scale_y, mode)

    # pad/crop
    vol_pad, pc_rec = pad_or_crop_xy(vol_xy, INTERMEDIATE_SHAPE)
    record["pad_crop_xy"] = pc_rec

    # 第二次縮放（若需要）
    if FINAL_SHAPE != INTERMEDIATE_SHAPE:
        second_scale = FINAL_SHAPE / INTERMEDIATE_SHAPE
        record["second_scale_xy"] = second_scale
        vol_out = resample_xy_slices(vol_pad, second_scale, second_scale, mode)
    else:
        record["second_scale_xy"] = 1.0
        vol_out = vol_pad

    return vol_out, record

# ---------------------------------------------------
# 主流程
# ---------------------------------------------------

img_files = sorted(glob(os.path.join(MRI_IMG_DIR, "*.nii.gz")))
lbl_files = sorted(glob(os.path.join(MRI_LBL_DIR, "*.nii.gz")))
assert len(img_files) == len(lbl_files)

spacing_records: Dict[str, Dict] = {}

# 以第一份影像的朝向作 reference
ref_axcodes = ("R", "A", "S")
ref_ornt = axcodes2ornt(ref_axcodes)

for idx, (img_p, lbl_p) in enumerate(zip(img_files, lbl_files), start=1):
    uid = f"{idx:04d}"

    # ---------- 讀取 ----------
    img_nii, lbl_nii = nib.load(img_p), nib.load(lbl_p)
    img_np, lbl_np = img_nii.get_fdata(), lbl_nii.get_fdata()
    orig_aff = img_nii.affine

    # ---------- 朝向對齊 ----------
    orig_axcodes = aff2axcodes(orig_aff)
    cur_ornt = axcodes2ornt(orig_axcodes)
    tfm = ornt_transform(cur_ornt, ref_ornt)

    img_np = apply_orientation(img_np, tfm).copy()
    lbl_np = apply_orientation(lbl_np, tfm).copy()
    aligned_aff = orig_aff @ nib.orientations.inv_ornt_aff(tfm, img_np.shape)

    # ---------- 計算 spacing ----------
    spacing_y = float(np.linalg.norm(aligned_aff[:3, 0]))
    spacing_x = float(np.linalg.norm(aligned_aff[:3, 1]))
    spacing_z = float(np.linalg.norm(aligned_aff[:3, 2]))

    # ---------- X/Y 預處理 ----------
    img_xy, rec_img_xy = preprocess_xy(img_np, spacing_x, spacing_y, "bilinear", True)
    lbl_xy, rec_lbl_xy = preprocess_xy(lbl_np, spacing_x, spacing_y, "nearest", False)

    # ---------- Z 重採樣 (在裁剪之前) ----------
    scale_z = spacing_z / TARGET_SPACING_Z
    img_xyz = resample_z(img_xy, scale_z, "bilinear")
    lbl_xyz = resample_z(lbl_xy, scale_z, "nearest")

    # ---------- 裁剪 / 補零 Z 軸 ----------
    img_fin, lbl_fin, z_rec = crop_or_pad_z(img_xyz, lbl_xyz, TARGET_SLICES_Z)

    # ---------- 生成新仿射矩陣 ----------
    new_aff = get_new_affine(orig_aff,
                             spacing_xy=TARGET_SPACING_XY,
                             spacing_z=TARGET_SPACING_Z,
                             z_shift=z_rec["crop_top"])

    # ---------- 保存 ----------
    nib.save(nib.Nifti1Image(img_fin, new_aff),
             os.path.join(MRI_OUT_DIR, f"image_{uid}.nii.gz"))
    nib.save(nib.Nifti1Image(lbl_fin, new_aff),
             os.path.join(MRI_OUT_DIR, f"label_{uid}.nii.gz"))

    # ---------- 記錄 ----------
    spacing_records[uid] = {
        "image_file": os.path.basename(img_p),
        "label_file": os.path.basename(lbl_p),
        "orig_axcodes": orig_axcodes,
        "target_axcodes": ref_axcodes,
        "ornt_transform": tfm.tolist(),
        "spacing_orig": {"x": spacing_x, "y": spacing_y, "z": spacing_z},
        "spacing_target": {"xy": TARGET_SPACING_XY, "z": TARGET_SPACING_Z},
        "scale_xy": rec_img_xy["scale_xy"],
        "scale_z": scale_z,
        "second_scale_xy": rec_img_xy["second_scale_xy"],
        "pad_crop_xy": {
            "img": rec_img_xy["pad_crop_xy"],
            "lbl": rec_lbl_xy["pad_crop_xy"],
        },
        "z_crop_pad": z_rec,
        "shape_after_xy": list(img_xy.shape),
        "shape_after_z_resample": list(img_xyz.shape),
        "shape_final": list(img_fin.shape),
        "orig_affine": orig_aff.tolist(),
        "aligned_affine": aligned_aff.tolist(),
        "output_affine": new_aff.tolist(),
        "original_shape": img_nii.shape,
    }

    print(f"✅ {uid}: save in {MRI_OUT_DIR}")
    print(z_rec)
    print(img_fin.shape)

# ---------------------------------------------------
# 保存 JSON 記錄
# ---------------------------------------------------
with open(SPACING_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(spacing_records, f, indent=2, ensure_ascii=False)

print(f"\n🎉 process{len(img_files)}  MRI ")
print(f"➡️  output_dir：{MRI_OUT_DIR}")
print(f"➡️  file：{SPACING_JSON_PATH}")












