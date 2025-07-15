import os
import torch
import torchvision
import random
import numpy as np
from PIL import Image

import cv2

import data.transform_utils as trans

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


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img * (min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img


# implementation by torchvision, detail in
# https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14


totensor = torchvision.transforms.ToTensor()
tr_func  = trans.transform_with_label(trans.tr_aug)

def transform_augment(img_list, nclass, sequence = None, split='val', min_max=(0, 1)):
    if split == 'train':
        
        image = img_list[0]
        label = img_list[1]
        
        comp = np.stack( [image, label], axis = -1 )
        image, label = tr_func(comp, c_img = 1, c_label = 1, nclass = nclass, is_train = True, use_onehot = False)
        
        image = np.float32(image)
        label = np.float32(label)
        
        imgs = [image, label]
        
        imgs = [totensor(img) for img in imgs]
        imgs = torch.stack(imgs, 0)
        imgs = torch.unbind(imgs, dim=0)
        
    elif split == 'val':

        image = img_list[0]
        label = img_list[1]

        image = np.float32(image)
        label = np.float32(label)

        imgs = [image, label]
        
        imgs = [totensor(img) for img in imgs]
        imgs = torch.stack(imgs, 0)
        imgs = torch.unbind(imgs, dim=0)
        
    return imgs
