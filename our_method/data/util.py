import os
import torchvision
import numpy as np
import data.transform_utils as trans
import torch.nn.functional as F
import torch

from .location_scale_augmentation import LocationScaleAugmentation

join = os.path.join
IMG_EXTENSIONS = ['.npy', '.nii.gz']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(dir, sequence):
    assert os.path.isdir(dir), '{:s} is not a valid directory'.format(dir)
    image_path = []
    if sequence == "all":
        idx = 0  # index of sequence of images
        for folder_name in sorted(os.listdir(dir)):
            img_dir = join(dir, folder_name)
            image_path.append([])
            for img_name in sorted(os.listdir(img_dir)):
                if img_name.endswith('.npy'):
                    image_path[idx].append(join(img_dir, img_name))
            idx += 1
    else:
        img_dir = join(dir, sequence)
        assert os.path.isdir(dir), '{:s} is not a valid name for sequence images'.format(sequence)
        for img_name in sorted(os.listdir(img_dir)):
            if is_image_file(img_name):
                image_path.append(join(img_dir, img_name))

    assert image_path, '{:s} has no valid image file'.format(dir)
    return sorted(image_path)



totensor = torchvision.transforms.ToTensor()
tr_func  = trans.transform_with_label(trans.tr_aug)

    
def to_tensor(arr, is_label=False):


    if isinstance(arr, torch.Tensor):
        return arr
    dtype = torch.float32 if not is_label else torch.int64
    return torch.from_numpy(arr).to(dtype)

def to_original(tensor, original_sample):

    if isinstance(original_sample, torch.Tensor):
        return tensor
    return tensor.cpu().numpy().astype(original_sample.dtype)

def transform_augment(img_list, nclass, thres=0.9, split='val'):
    """根据 split (train / val) 对输入做增广与预处理。

    在 `train` 和 `val` 两个阶段均统一执行 320×320 的 resize 逻辑，
    保持尺寸处理方式一致，减少推理阶段与训练阶段的分布差异。
    """

    # 公共配置
    img_pad_val = 0.0        # 若后续需 pad，可在此处修改
    lbl_pad_val = 255        # 分割任务常用 ignore_index
    target_size = (320, 320) # 统一缩放尺寸
    bd_bias = 32

    # ---------------------------
    # —— 训练阶段 ——————————————————
    # ---------------------------
    if split == 'train':
        ori_image         = img_list[0]
        ori_label         = img_list[1]
        pre_pesudolabel   = img_list[2]
        ori_mask          = img_list[3]

        ori_pesudolabel = prepare_pesudolabel(pre_pesudolabel, thres)

        # ---------- resize 部分（保留原先逻辑） ----------
        if not isinstance(ori_image, torch.Tensor):
            image       = torch.from_numpy(ori_image)
            label       = torch.from_numpy(ori_label)
            pesudolabel = torch.from_numpy(ori_pesudolabel)
            mask        = torch.from_numpy(ori_mask)

        # image: (H, W, 3) → (1, 3, H, W) → resize → (H, W, 3)
        image = image[bd_bias:-bd_bias, bd_bias:-bd_bias, :]
        image = image.permute(2, 0, 1).unsqueeze(0).float()
        image = F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)[0]
        image = image.permute(1, 2, 0)

        # label: (H, W) → (1, 1, H, W) → resize → (H, W)
        label = label[bd_bias:-bd_bias, bd_bias:-bd_bias]
        label = label.unsqueeze(0).unsqueeze(0).float()
        label = F.interpolate(label, size=target_size, mode='nearest')[0, 0].long()
        
        pesudolabel = pesudolabel[bd_bias:-bd_bias, bd_bias:-bd_bias]
        pesudolabel = pesudolabel.unsqueeze(0).unsqueeze(0).float()
        pesudolabel = F.interpolate(pesudolabel, size=target_size, mode='nearest')[0, 0].long()

        # mask: (H, W, 3) 跟随 image 的 resize 策略
        mask = mask[bd_bias:-bd_bias, bd_bias:-bd_bias, :]
        mask = mask.permute(2, 0, 1).unsqueeze(0).float()
        mask = F.interpolate(mask, size=target_size, mode='nearest')[0]
        mask = mask.permute(1, 2, 0)

        # 转回 numpy, 与后续逻辑保持一致
        image       = to_original(image, ori_image)
        label       = to_original(label, ori_label)
        pesudolabel = to_original(pesudolabel, ori_label)
        mask        = to_original(mask, ori_mask)

        # ================= 后续增广逻辑保持不变 =================
        vmax, vmin = image.max(), image.min()
        image = (image - vmin) / (vmax - vmin + 1e-8)

        location_scale1 = LocationScaleAugmentation(vrange=(0., 1.), background_threshold=0.01)
        GLA1 = location_scale1.Global_Location_Scale_Augmentation(image.copy())
        GLA1 = GLA1 * (vmax - vmin) + vmin
        LLA1 = location_scale1.Local_Location_Scale_Augmentation(image.copy(), mask.astype(np.int32))
        LLA1 = LLA1 * (vmax - vmin) + vmin

        location_scale2 = LocationScaleAugmentation(vrange=(0., 1.), background_threshold=0.01)
        GLA2 = location_scale2.Global_Location_Scale_Augmentation(image.copy())
        GLA2 = GLA2 * (vmax - vmin) + vmin
        LLA2 = location_scale2.Local_Location_Scale_Augmentation(image.copy(), mask.astype(np.int32))
        LLA2 = LLA2 * (vmax - vmin) + vmin

        comp = np.stack([
            GLA1[..., 0], GLA1[..., 1], GLA1[..., 2],
            LLA1[..., 0], LLA1[..., 1], LLA1[..., 2],
            GLA2[..., 0], GLA2[..., 1], GLA2[..., 2],
            LLA2[..., 0], LLA2[..., 1], LLA2[..., 2],
            label,
            pesudolabel
        ], axis=-1)

        timg1, timg2, label, pesudolabel = tr_func(
            comp,
            c_img=12,
            c_label=1,
            c_plabel=1,
            nclass=nclass,
            is_train=True,
            use_onehot=False,
        )
        GLA1, LLA1 = np.split(timg1, 2, -1)
        GLA2, LLA2 = np.split(timg2, 2, -1)

        img1, aug_img1 = np.float32(GLA1), np.float32(LLA1)
        img2, aug_img2 = np.float32(GLA2), np.float32(LLA2)
        label = np.int64(label)
        pesudolabel = np.float32(pesudolabel)

        imgs = [img1, aug_img1, img2, aug_img2, label, pesudolabel]
        imgs = [totensor(img) for img in imgs]

    # ---------------------------
    # —— 验证阶段 ——————————————————
    # ---------------------------
    elif split == 'val':
        # 先保留原始引用，方便 to_original 还原 numpy 格式
        ori_image         = img_list[0]
        ori_label         = img_list[1]
        pre_pesudolabel   = img_list[2]
        ori_mask          = img_list[3]
        ori_pesudolabel = pre_pesudolabel

        # ---------- 新增：复用训练阶段的 resize 逻辑 ----------
        if not isinstance(ori_image, torch.Tensor):
            image       = torch.from_numpy(ori_image)
            label       = torch.from_numpy(ori_label)
            pesudolabel = torch.from_numpy(ori_pesudolabel)
            mask        = torch.from_numpy(ori_mask)

        image = image.permute(2, 0, 1).unsqueeze(0).float()
        image = F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)[0]
        image = image.permute(1, 2, 0)

        label = label.unsqueeze(0).unsqueeze(0).float()
        label = F.interpolate(label, size=target_size, mode='nearest')[0, 0].long()

        pesudolabel = pesudolabel.unsqueeze(0).unsqueeze(0).float()
        pesudolabel = F.interpolate(pesudolabel, size=target_size, mode='nearest')[0, 0].long()
        


        mask = mask.permute(2, 0, 1).unsqueeze(0).float()
        mask = F.interpolate(mask, size=target_size, mode='nearest')[0]
        mask = mask.permute(1, 2, 0)

        # 转回 numpy (保持 API 一致)
        image       = to_original(image, ori_image)
        label       = to_original(label, ori_label)
        pesudolabel = to_original(pesudolabel, ori_label)  # 与 label 保持同格式
        mask        = to_original(mask, ori_mask)
        
        

        # 验证阶段无需复杂增广，直接生成四份同样的图像，保持接口兼容
        img1 = aug_img1 = img2 = aug_img2 = np.float32(image)
        label = np.int64(label)
        pesudolabel = np.float32(pesudolabel)

        imgs = [img1, aug_img1, img2, aug_img2, label, pesudolabel]
        imgs = [totensor(img) for img in imgs]
        
    elif split == 'test':
        # 先保留原始引用，方便 to_original 还原 numpy 格式
        ori_image         = img_list[0]
        ori_label         = img_list[1]
        pre_pesudolabel   = img_list[2]
        ori_mask          = img_list[3]
        ori_pesudolabel = prepare_pesudolabel(pre_pesudolabel, thres)

        # ---------- 新增：复用训练阶段的 resize 逻辑 ----------
        if not isinstance(ori_image, torch.Tensor):
            image       = torch.from_numpy(ori_image)
            label       = torch.from_numpy(ori_label)
            pesudolabel = torch.from_numpy(ori_pesudolabel)
            mask        = torch.from_numpy(ori_mask)
        image = image[bd_bias:-bd_bias, bd_bias:-bd_bias, :]
        image = image.permute(2, 0, 1).unsqueeze(0).float()
        image = F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)[0]
        image = image.permute(1, 2, 0)
        label = label[bd_bias:-bd_bias, bd_bias:-bd_bias]
        label = label.unsqueeze(0).unsqueeze(0).float()
        label = F.interpolate(label, size=target_size, mode='nearest')[0, 0].long()
        pesudolabel = pesudolabel[bd_bias:-bd_bias, bd_bias:-bd_bias]
        pesudolabel = pesudolabel.unsqueeze(0).unsqueeze(0).float()
        pesudolabel = F.interpolate(pesudolabel, size=target_size, mode='nearest')[0, 0].long()
        
        mask = mask[bd_bias:-bd_bias, bd_bias:-bd_bias, :]

        mask = mask.permute(2, 0, 1).unsqueeze(0).float()
        mask = F.interpolate(mask, size=target_size, mode='nearest')[0]
        mask = mask.permute(1, 2, 0)

        # 转回 numpy (保持 API 一致)
        image       = to_original(image, ori_image)
        label       = to_original(label, ori_label)
        pesudolabel = to_original(pesudolabel, ori_label)  # 与 label 保持同格式
        mask        = to_original(mask, ori_mask)

        # 验证阶段无需复杂增广，直接生成四份同样的图像，保持接口兼容
        img1 = aug_img1 = img2 = aug_img2 = np.float32(image)
        label = np.int64(label)
        pesudolabel = np.float32(pesudolabel)

        imgs = [img1, aug_img1, img2, aug_img2, label, pesudolabel]
        imgs = [totensor(img) for img in imgs]

    else:
        raise ValueError(f"Unsupported split: {split}")

    return imgs
    


def prepare_pesudolabel(pre_pesudolabel, thres):

    pre_pesudolabel = np.squeeze(pre_pesudolabel)
    c, h, w = np.shape(pre_pesudolabel)

    # define an all-zero volume for output.
    pesudolabel = np.zeros((h, w))

    # for each pixel in (jh, kw).
    for jh in range(h):
        for kw in range(w):

            classVec = np.arange(c-1) + 1  # (c-1)-length class vector, without considering background in class-0.

            # for each forground class in 1-to-c.
            for iclass in classVec:

                # if the probability larger than the thres, then take it;
                #   otherwise, skip to next pixel.
                if pre_pesudolabel[iclass, jh, kw] > thres:
                    pesudolabel[jh, kw] = iclass

    return pesudolabel
