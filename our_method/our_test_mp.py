import time

import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
import re

join = os.path.join

if __name__ == "__main__":
    # print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/Ours_v2_train.json', help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'], default='train', help='Run either train(training) or val(generation)')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    # args.phase = 'val'
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        import wandb

        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']
    n_epoch = opt['train']['n_epoch']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(current_epoch, current_step))



    num_classes = 14
    logger.info('Begin Model Evaluation.')
    avg_dice = np.zeros(num_classes - 1,)

    idx = 0
    result_dir = '../evaluation'
    os.makedirs(result_dir, exist_ok=True)
    data_info = np.genfromtxt('../dataset/slice_info.txt',
                          delimiter='\t',     # 按制表符分隔；若是空格可去掉
                          names=True,         # 让第一行当列名
                          dtype=None,         # 自动推断类型
                          encoding=None)      # 读文本文件推荐加这一行

    # 现在返回的是结构化数组，可直接用列名访问
    val = data_info['slice_num']
    slice_num = val.tolist() if hasattr(val, "tolist") and val.shape else [val]

    for _, val_data in enumerate(val_loader):
        
        start_time = time.time()
        
        idx += 1

        GLA_img1 = val_data['GLA1']
        LLA_img1 = val_data['LLA1']
        GLA_img2 = val_data['GLA2']
        LLA_img2 = val_data['LLA2']
        lbl      = val_data['target']
        plabel   = val_data['pesudolabel']
        Index    = val_data['Index']
        


        bs_size, c, w, h = GLA_img1.shape

        input_data = {'ori_aug1': GLA_img1.view(bs_size, -1),
                      'ori_aug2': GLA_img2.view(bs_size, -1),
                      'target': lbl.view(bs_size, -1),
                      'pesudolabel': plabel.view(bs_size, -1),
                      'Index': Index}

        # bs_size = opt['datasets']['val']['batch_size']

        diffusion.feed_data(input_data)
        
        val_id_acc = Index[0]
        path = val_set.ori_path[val_id_acc]
        image_name = os.path.basename(path)
        match = re.search(r'MR-id(\d+)-', image_name)
        if match:
            id_str = match.group(1)  # 提取到的是字符串 '01'
            id_num = int(id_str)     # 转换为整数 1
        else:
            print("未找到匹配的ID")
            
        if slice_num[id_num-1] < 200:
            diffusion.generate_content(diffusion.data,
                         batch_size=bs_size,
                         filter_ratio=0.,
                         sample_type="top0.85r,fast200",
                         )
        elif slice_num[id_num-1] >= 200 and slice_num[id_num-1] <=400:
            diffusion.generate_content(diffusion.data,
                         batch_size=bs_size,
                         filter_ratio=0.,
                         sample_type="top0.85r,fast500",
                         )
        else:
            diffusion.generate_content(diffusion.data,
                         batch_size=bs_size,
                         filter_ratio=0.,
                         sample_type="top0.85r,fast1000",
                         )

        end_time = time.time()
        print("Runtime:" + str(end_time - start_time) + " sec")

        visuals = diffusion.get_out_visuals()

        img_size = opt['model']['diffusion_config']['embed_opt']['img_size']
        k_slice = opt['model']['diffusion_config']['embed_opt']['initconv_in_chans']

        ori_aug1 = Metrics.tensor2img_mp(visuals['ori_aug1'].view(bs_size, k_slice, img_size, img_size), min_max=(-10, 10))
        ori_aug1_int = (ori_aug1 * 255.0).round()
        est_label_aug1 = Metrics.tensor2img_mp(visuals['est_label_aug1'].view(bs_size, img_size, img_size), min_max=(0, num_classes))
        est_label_aug1_int = (est_label_aug1 * 255.0).round()

        ori_aug2 = Metrics.tensor2img_mp(visuals['ori_aug2'].view(bs_size, k_slice, img_size, img_size), min_max=(-10, 10))
        ori_aug2_int = (ori_aug2 * 255.0).round()
        est_label_aug2 = Metrics.tensor2img_mp(visuals['est_label_aug2'].view(bs_size, img_size, img_size), min_max=(0, num_classes))
        est_label_aug2_int = (est_label_aug2 * 255.0).round()

        gt_label = Metrics.tensor2img_mp(visuals['gt_label'].view(bs_size, img_size, img_size), min_max=(0, num_classes))
        gt_label_int = (gt_label * 255.0).round()
        pesudolabel = Metrics.tensor2img_mp(visuals['pesudolabel'].view(bs_size, img_size, img_size), min_max=(0, num_classes))
        pesudolabel_int = (pesudolabel * 255.0).round()
        
        avg_dice_fusion = np.zeros(num_classes - 1,)
        
        
        


        for ibatch in range(bs_size):

            tmp0 = np.concatenate((ori_aug1_int[ibatch, :, :, 1],
                                   est_label_aug1_int[ibatch, :, :],
                                   gt_label_int[ibatch, :, :]), axis=1)
            tmp1 = np.concatenate((ori_aug2_int[ibatch, :, :, 1],
                                   est_label_aug2_int[ibatch, :, :],
                                   pesudolabel_int[ibatch, :, :]), axis=1)
            visual_img = np.concatenate((tmp0, tmp1), axis=0)

            val_id = val_data['Index'][ibatch]
            path = val_set.ori_path[val_id]
            image_name = os.path.basename(path)

            # generation
            Metrics.save_img(visual_img, '{}/{}_fake.png'.format(result_dir, image_name))

            #print('{}/epoch{:0>3d}_fake_{}'.format(result_dir, current_epoch, image_name))

            # generation
            #np.save('{}/{}_fake1.npy'.format(result_dir, image_name), est_label_aug1[ibatch, :, :] * num_classes)
            #np.save('{}/{}_fake2.npy'.format(result_dir, image_name), est_label_aug2[ibatch, :, :] * num_classes)
            #np.save('{}/{}_ori.npy'.format(result_dir, image_name), ori_aug1[ibatch, :, :, 1])
            #np.save('{}/{}_tg.npy'.format(result_dir, image_name), gt_label[ibatch, :, :] * num_classes)
            #np.save('{}/{}_pl.npy'.format(result_dir, image_name), pesudolabel[ibatch, :, :] * num_classes)

            # compute validation score.
            avg_dice += Metrics.calculate_dice(gt_label[ibatch, :, :], est_label_aug1[ibatch, :, :], num_classes)
            
           
            label_dtype = (est_label_aug1[ibatch, :, :] * num_classes).dtype
            priority = np.array([1, 2, 13, 5, 3, 6, 11, 4, 9, 10, 8, 12, 7], dtype=label_dtype)
            fused_label = np.zeros_like(est_label_aug1[ibatch, :, :])
            for cls in priority:               
                fused_label[est_label_aug1[ibatch, :, :]* num_classes == cls] = cls
                fused_label[est_label_aug2[ibatch, :, :]* num_classes == cls] = cls
            np.save(f'{result_dir}/{image_name}_fakeFusion.npy', fused_label)


        final_time = time.time()
        

