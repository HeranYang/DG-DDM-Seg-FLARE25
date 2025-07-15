import nibabel as nib
import numpy as np
import math
import glob
import os

##======================================
data_num = 50
idx_pct = [1, 0, 0]
LIR, HIR = -125, 275  # 截断范围

tr_size     = round(data_num * idx_pct[0])
val_size    = math.floor(data_num * idx_pct[1])
te_size     = data_num - tr_size - val_size
##======================================

#root_path = r'/home/hryang/Project/DMSeg_Project/data/multi-modality-data/std_processed_data/SABSCT/'
#input_image_name = root_path + r'processed/image_{}.nii.gz'
#input_label_name = root_path + r'processed/label_{}.nii.gz'
##======================================

root_image_path = r'/data/FLARE_Challenge/data/original_data/train_CT_gt_label/imagesTr/'
root_label_path = r'/data/FLARE_Challenge/data/original_data/train_CT_gt_label/labelsTr/'
input_image_name = root_image_path + 'FLARE22_Tr_{}_0000.nii.gz'
input_label_name = root_label_path + 'FLARE22_Tr_{}.nii.gz'

#save_path = r'/home/hryang/Project/DMSeg_Project/data/multi-modality-data/processing_dataset/step2_saveSlice/CT_to_MR/'

save_path = r'/data/FLARE_Challenge/data/multi-modality-data/processing_dataset/step2_saveSlice/CT_to_MR/'
train_folder = save_path + r'train/'
valid_folder = save_path + r'valid/'
test_folder = save_path + r'test_ontrain/'

folders_to_create = [
    train_folder + 'image',
    train_folder + 'label',
    valid_folder + 'image',
    valid_folder + 'label',
    test_folder + 'image',
    test_folder + 'label',
]

for folder in folders_to_create:
    os.makedirs(folder, exist_ok=True)

#img_pids   = sorted([ fid.split("_")[-1].split(".nii.gz")[0] for fid in glob.glob(root_path  + "/processed/image_*.nii.gz") ], key = lambda x: int(x))

img_pids = sorted([
    fid.split("FLARE22_Tr_")[1].split("_0000.nii.gz")[0]
    for fid in glob.glob(root_image_path + "FLARE22_Tr_*_0000.nii.gz")
])
te_ids     = img_pids[: te_size]
val_ids    = img_pids[te_size: te_size + val_size]
tr_ids     = img_pids[te_size + val_size: ]


##===================================================================================================================
# Estimate meanval and global_std on train subset.
print('Estimate mean and var on train set \n')

total_val = 0
n_pix = 0
for subid in tr_ids:
    in_img = nib.load(input_image_name.format(subid))
    in_img_vol = in_img.get_fdata().astype('single')
    in_img_vol = np.float32(in_img_vol)
    in_img_vol = np.clip(in_img_vol, LIR, HIR)

    total_val += in_img_vol.sum()
    n_pix += np.prod(in_img_vol.shape)
meanval = total_val / n_pix

total_var = 0
for subid in tr_ids:
    in_img = nib.load(input_image_name.format(subid))
    in_img_vol = in_img.get_fdata().astype('single')
    in_img_vol = np.float32(in_img_vol)
    in_img_vol = np.clip(in_img_vol, LIR, HIR)

    total_var += np.sum((in_img_vol - meanval) ** 2 )
var_all = total_var / n_pix
global_std = var_all ** 0.5

print("meanval on trainset = {} \n".format(meanval))
print("global_std on trainset = {} \n".format(global_std))
##===================================================================================================================


##===================================================================================================================
# Train subset.
for subid in tr_ids:

    print('processing Train Subject: id {} \n'.format(subid))

    input_image = nib.load(input_image_name.format(subid))
    input_image_vol = input_image.get_fdata().astype('single')
    input_image_vol = np.float32(input_image_vol)
    input_image_vol = np.transpose(input_image_vol, (1, 0, 2))
    input_image_vol = np.clip(input_image_vol, LIR, HIR)
    input_image_vol_normed = (input_image_vol - meanval) / global_std

    input_label = nib.load(input_label_name.format(subid))
    input_label_vol = input_label.get_fdata()
    input_label_vol = np.array(input_label_vol)
    input_label_vol = np.transpose(input_label_vol, (1, 0, 2))
    

    slice_num = input_image_vol.shape[2]
    slice_ids = np.arange(slice_num)

    save_image_name = train_folder + r'image/CT-id{:0>2d}-slice{:0>2d}.npy'
    save_label_name = train_folder + r'label/CT-id{:0>2d}-slice{:0>2d}.npy'

    for sliceid in slice_ids:

        np.save(save_image_name.format(int(subid), sliceid), input_image_vol_normed[:, :, sliceid])
        np.save(save_label_name.format(int(subid), sliceid), input_label_vol[:, :, sliceid])
##===================================================================================================================


##===================================================================================================================
# Valid subset.
for subid in val_ids:

    print('processing Valid Subject: id {} \n'.format(subid))

    input_image = nib.load(input_image_name.format(subid))
    input_image_vol = input_image.get_fdata().astype('single')
    input_image_vol = np.float32(input_image_vol)
    input_image_vol = np.transpose(input_image_vol, (1, 0, 2))
    input_image_vol = np.clip(input_image_vol, LIR, HIR)
    input_image_vol_normed = (input_image_vol - meanval) / global_std

    input_label = nib.load(input_label_name.format(subid))
    input_label_vol = input_label.get_fdata()
    input_label_vol = np.array(input_label_vol)
    input_label_vol = np.transpose(input_label_vol, (1, 0, 2))


    slice_num = input_image_vol.shape[2]
    slice_ids = np.arange(slice_num)

    save_image_name = valid_folder + r'image/CT-id{:0>2d}-slice{:0>2d}.npy'
    save_label_name = valid_folder + r'label/CT-id{:0>2d}-slice{:0>2d}.npy'

    for sliceid in slice_ids:

        np.save(save_image_name.format(int(subid), sliceid), input_image_vol_normed[:, :, sliceid])
        np.save(save_label_name.format(int(subid), sliceid), input_label_vol[:, :, sliceid])
##===================================================================================================================


##===================================================================================================================
# Test subset.
for subid in te_ids:

    print('processing Test Subject: id {} \n'.format(subid))

    input_image = nib.load(input_image_name.format(subid))
    input_image_vol = input_image.get_fdata().astype('single')
    input_image_vol = np.float32(input_image_vol)
    input_image_vol = np.transpose(input_image_vol, (1, 0, 2))
    input_image_vol = np.clip(input_image_vol, LIR, HIR)
    input_image_vol_normed = (input_image_vol - meanval) / global_std

    input_label = nib.load(input_label_name.format(subid))
    input_label_vol = input_label.get_fdata()
    input_label_vol = np.array(input_label_vol)
    input_label_vol = np.transpose(input_label_vol, (1, 0, 2))


    slice_num = input_image_vol.shape[2]
    slice_ids = np.arange(slice_num)

    save_image_name = test_folder + r'image/CT-id{:0>2d}-slice{:0>2d}.npy'
    save_label_name = test_folder + r'label/CT-id{:0>2d}-slice{:0>2d}.npy'

    for sliceid in slice_ids:

        np.save(save_image_name.format(int(subid), sliceid), input_image_vol_normed[:, :, sliceid])
        np.save(save_label_name.format(int(subid), sliceid), input_label_vol[:, :, sliceid])
##===================================================================================================================
