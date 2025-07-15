# MRI_dataset.py
import os
import random
import re
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import data.util as Util


class MRIDataset(Dataset):
    """加载 3-slice CT/MR 影像 + GT + 伪标签的数据集。

    Args:
        dataroot (str): 数据根目录
        data_sequence (dict): 由配置文件给出的子文件夹映射
        nclass (int): 类别数
        thres (float): transform_augment 中使用的阈值
        randnum_pl (int): 每张 slice 对应的伪标签文件夹数量
        split (str): 'train' | 'val' | 'test'
        data_len (int): 若 >0 则截断数据集长度
    """

    _SLICE_PATTERN = re.compile(r"slice(\d+)\.npy$")

    def __init__(
        self,
        dataroot: str,
        data_sequence: dict,
        nclass: int,
        thres: float,
        randnum_pl: int,
        split: str = "train",
        data_len: int = -1,
    ):
        super().__init__()

        self.data_sequence = data_sequence
        self.split = split
        self.nclass = nclass
        self.thres = thres
        self.randnum_pl = randnum_pl

        self.ori_path = Util.get_paths_from_images(dataroot, data_sequence["cond_image"])
        self.tg_path = Util.get_paths_from_images(dataroot, data_sequence["output_label"])
        self.pl_path = Util.get_paths_from_images(dataroot, data_sequence["cond_pesudolabel"])

        self.dataset_len = len(self.ori_path)
        self.data_len = self.dataset_len if data_len <= 0 else min(data_len, self.dataset_len)

    # ---------------------------------------------------------------------- #
    #  Required by torch.utils.data.Dataset
    # ---------------------------------------------------------------------- #
    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int) -> dict:
        # ------------------------------------------------------------------ #
        # 1. 构造三张相邻 slice 的文件名
        # ------------------------------------------------------------------ #
        cur_img_path = self.ori_path[index]
        slice_triplet = self._get_neighbor_triplet(cur_img_path)  # (prev, cur, next)

        # ------------------------------------------------------------------ #
        # 2. 加载 3-slice 影像
        # ------------------------------------------------------------------ #
        ori_slices = [self._safe_load(p) for p in slice_triplet]
        ori = np.stack(ori_slices, axis=-1)  # (H, W, 3)

        # ------------------------------------------------------------------ #
        # 3. 加载 GT 与伪标签
        # ------------------------------------------------------------------ #
        tg_path_triplet = self._replace_root(cur_img_path, self.tg_path[index], slice_triplet)
        tg_slices = [self._safe_load(p) for p in tg_path_triplet]
        tg = tg_slices[1]                                           # 当前 slice 的 GT
        mask = np.stack(tg_slices, axis=-1)                         # 三 slice GT 供 LLA

        pl_path_triplet = self._replace_root(
            cur_img_path,
            self.pl_path[index],
            slice_triplet,
            pesudo=True,
        )
        pre_pl = self._safe_load(pl_path_triplet[1])                # 只要中心 slice 的伪标签

        # ------------------------------------------------------------------ #
        # 4. 数据增广
        # ------------------------------------------------------------------ #
        GLA1, LLA1, GLA2, LLA2, tg, pl = Util.transform_augment(
            [ori, tg, pre_pl, mask],
            self.nclass,
            thres=self.thres,
            split=self.split,
        )

        return {
            "GLA1": GLA1,
            "LLA1": LLA1,
            "GLA2": GLA2,
            "LLA2": LLA2,
            "target": tg.to(torch.int64),
            "pesudolabel": pl,
            "Index": index,
        }

    # ------------------------------------------------------------------ #
    #  Helper functions
    # ------------------------------------------------------------------ #
    def _parse_slice_id(self, fname: str) -> Tuple[int, int]:
        """返回 (slice_id, 位数)。若匹配失败抛异常。"""
        m = self._SLICE_PATTERN.search(fname)
        if m is None:
            raise ValueError(f"非法文件名，未找到 sliceID: {fname}")
        slice_id_str = m.group(1)
        return int(slice_id_str), len(slice_id_str)

    def _build_filename(self, head: str, slice_id: int, width: int, tail: str) -> str:
        """按原始位宽生成新的文件名。"""
        return f"{head}{slice_id:0{width}d}{tail}"

    def _get_neighbor_triplet(self, img_path: str) -> List[str]:
        """返回当前/前一/后一 slice 的完整路径列表。"""
        dir_path, fname = os.path.split(img_path)
        head, tail = fname.split("slice")[0] + "slice", ".npy"
        slice_id, width = self._parse_slice_id(fname)

        def exists(f: str) -> bool:
            return os.path.exists(os.path.join(dir_path, f))

        # center slice
        cur_name = self._build_filename(head, slice_id, width, tail)

        # previous slice
        if slice_id == 0:
            prev_name = cur_name
        else:
            prev_tmp = self._build_filename(head, slice_id - 1, width, tail)
            prev_name = prev_tmp if exists(prev_tmp) else cur_name

        # next slice
        next_tmp = self._build_filename(head, slice_id + 1, width, tail)
        next_name = next_tmp if exists(next_tmp) else cur_name

        return [os.path.join(dir_path, n) for n in (prev_name, cur_name, next_name)]

    def _safe_load(self, path: str) -> np.ndarray:
        """若文件不存在，则尝试回退到父目录同名文件；仍失败则抛错。"""
        if os.path.exists(path):
            return np.load(path)

        # 最后的兜底：用中心 slice（即 dirname 相同、slice id 不变）
        dirname, fname = os.path.split(path)
        center_path = os.path.join(dirname, fname)
        if os.path.exists(center_path):
            return np.load(center_path)

        raise FileNotFoundError(f"无法找到文件: {path}")

    def _replace_root(
        self,
        ref_ori_path: str,
        ref_target_path: str,
        triplet: List[str],
        pesudo: bool = False,
    ) -> List[str]:
        """把三张切片的路径从原图目录换成 GT/伪标签目录。

        Args:
            ref_ori_path: 数据集中存的原图路径（用来确定原目录结构）
            ref_target_path: 对应的 gt 或 pesudo 路径（用来确定目标根目录）
            triplet: 原图 3-slice 完整路径
            pesudo: 如果是伪标签，需要随机选子文件夹
        """
        ori_root = os.path.dirname(ref_ori_path)
        tgt_root = os.path.dirname(ref_target_path)

        if pesudo:
            # /pesudolabel/xx/ → 重新随机一个子文件夹
            tgt_root = os.path.dirname(tgt_root)
            rand_id = random.randint(1, self.randnum_pl)
            tgt_root = os.path.join(tgt_root, str(rand_id))

        out_paths = []
        for p in triplet:
            relative = os.path.relpath(p, ori_root)  # e.g. CT-id35-slice012.npy
            out_paths.append(os.path.join(tgt_root, relative))
            
        for p in out_paths:
            if not os.path.exists(p):
                print(f"[警告] 伪标签路径不存在: {p}")
        return out_paths

