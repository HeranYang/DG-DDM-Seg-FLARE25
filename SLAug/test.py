import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append(os.getcwd())
import argparse
from torch.utils.data import DataLoader
import numpy as np
import glob
from omegaconf import OmegaConf
from main import instantiate_from_config
import torch

import time
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-s",
        "--save_pseudo_dir",
        type=str,
        required=True,
        nargs="?",
        help="Path to save pseudo labels, e.g., ./data/pseudo_labels/",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-c",
        "--config",
        nargs="?",
        metavar="single_config.yaml",
        help="path to single config. If specified, base configs will be ignored "
        "(except for the last one if left unspecified).",
        const=True,
        default="",
    )
    parser.add_argument(
        "--ignore_base_data",
        action="store_true",
        help="Ignore data specification from base configs. Useful if you want "
        "to specify a custom datasets on the command line.",
    )
    return parser

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    parser = get_parser()
    opt, unknown = parser.parse_known_args()

    ckpt = None
    save_path = opt.save_pseudo_dir
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            try:
                idx = len(paths)-paths[::-1].index("logs")+1
            except ValueError:
                idx = -2 # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            for f in os.listdir(os.path.join(logdir,"checkpoints")):
                if 'val_best_epoch_2' in f:
                    ckpt = os.path.join(logdir, "checkpoints", f)
        print(f"logdir:{logdir}")
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*-project.yaml")))
        opt.base = base_configs+opt.base

    if opt.config:
        if type(opt.config) == str:
            opt.base = [opt.config]
        else:
            opt.base = [opt.base[-1]]

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)
    model_config = config.pop("model", OmegaConf.create())
    print(model_config)

    gpu = True
    eval_mode = True
    show_config = False
    model = instantiate_from_config(model_config)
    pl_sd=torch.load(ckpt, map_location="cpu")
    model.load_state_dict(pl_sd['model'], strict=False)
    model.cuda().eval()

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    val_loader = DataLoader(data.datasets["validation"], batch_size=1, num_workers=1)
    
    
    
    #test_loader = DataLoader(data.datasets["test"], batch_size=1, num_workers=1)
    label_name=[
        "bg",                  # 0
        "liver",               # 1
        "right_kidney",        # 2
        "spleen",              # 3
        "pancreas",            # 4
        "aorta",               # 5
        "inferior_vena_cava",  # 6
        "right_adrenal_gland", # 7
        "left_adrenal_gland",  # 8
        "gallbladder",         # 9
        "esophagus",           #10
        "stomach",             #11
        "duodenum",            #12
        "left_kidney"          #13
    ]
        

    # ========== 2. 设置路径 ==========
    image_save_dir = os.path.join(save_path, "image")
    label_save_dir = os.path.join(save_path, "label")
    pesudolabel_save_dir = os.path.join(save_path, "pesudolabel/1")
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)
    os.makedirs(pesudolabel_save_dir, exist_ok=True)
    
    slice_info_records = []
    curr_slice_count = 0

    # ========== 3. 开始推理 ==========
    model.eval()

    with torch.no_grad():
        volume_index = 0   

        for batch in val_loader:
            if batch['is_start']:
                volume_index += 1
                curr_slice_count = 0
                scan_id = batch['scan_id'][0]
                volume_start_time = time.time()
            img = batch['images'].cuda()         # [B, C, H, W]
            gth = batch['labels']
            pred = model(img)                   # [B, num_classes, H, W]
            pred = torch.argmax(pred, dim=1)    # [B, H, W]
            
            file_name = f"MR-id{volume_index:02d}-slice{curr_slice_count:02d}.npy"
            np.save(os.path.join(image_save_dir, file_name), img[0].squeeze().cpu().numpy())
            np.save(os.path.join(label_save_dir, file_name), gth[0].squeeze().cpu().numpy())
            np.save(os.path.join(pesudolabel_save_dir, file_name), pred[0].squeeze().cpu().numpy())


            curr_slice_count += 1
            
            if batch['is_end']:
                volume_end_time = time.time()
                elapsed = volume_end_time - volume_start_time
                print(f"Volume {volume_index:02d} ({scan_id}) done in {elapsed:.2f} sec")

                # 记录一条信息
                slice_info_records.append((volume_index, curr_slice_count))
            
        slice_info_txt = os.path.join(save_path, 'slice_info.txt')
        with open(slice_info_txt, 'w') as f:
            f.write('id\tslice_num\n')
            for vol_id, num_slices in slice_info_records:
                f.write(f"{vol_id}\t{num_slices}\n")
        print(f"✅ slice info saved to: {slice_info_txt}")
        

