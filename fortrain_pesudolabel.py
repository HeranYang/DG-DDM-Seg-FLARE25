import numpy as np
import os
import glob

##======================== bSSFP ============================
input_rootpath = r'/data/FLARE_Challenge/code/EarlyStopforpseudolabel/experiments/SR3 for MRI_250528_023132_image_label/results/'
input_folderlist = next(os.walk(input_rootpath))[1]
input_folderlist = sorted(input_folderlist)
folderNum = len(input_folderlist)

output_rootpath = r'/data/FLARE_Challenge/data/multi-modality-data/processing_dataset/step2_saveSlice/CT_to_MR/train/pesudolabel_{}/'.format(folderNum)
if not os.path.exists(output_rootpath):
    os.makedirs(output_rootpath)

index = 0
for input_folder in input_folderlist:

    print('processing folder: {} \n'.format(input_folder))

    index = index + 1
    output_path = output_rootpath + r'{}/'.format(index)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    input_path = os.path.join(input_rootpath, input_folder)
    input_image_all = sorted(glob.glob(input_path + "/*.npy_softfake.npy"))

    for input_image in input_image_all:

        image = np.load(input_image)

        (filepath, filename) = os.path.split(input_image)
        namesplit = filename.split('_')
        output_name = namesplit[0]

        save_image_name = output_path + output_name
        np.save(save_image_name, image)
