import torch
import data as Data
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np

from segmodel import create_forward

join = os.path.join

if __name__ == "__main__":
    # print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/SR3_EffNet_test.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
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
            train_loader = Data.create_dataloader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')
    

    time_vec = ["053457", "070814"]

    epoch_vec = [6,7,8,9,10]
    
    for time_id in time_vec:

        for epoch_id in epoch_vec:
    
            reload_model_fid = "experiments/SR3 for MRI_250527_{}_image_label/checkpoint/{}_net_Seg.pth".format(time_id, epoch_id)
            opt["segnet"]["reload_model_fid"] = reload_model_fid

            # ## ===============================================================================
            # # Segmentation Network Definition.
            # # model
            segnet = create_forward(opt['segnet'])
            logger.info('Initial SegNet Finished Epoch{}'.format(epoch_id))
            # ## ===============================================================================

            current_step = 0
            current_epoch = 0

            logger.info('Begin Model Evaluation.')

            avg_val_errors = 0.0
            idx = 0
            result_dir = os.path.join(opt['path']['results'], 'evaluation_time{}_epoch{}'.format(time_id, epoch_id))
            os.makedirs(result_dir, exist_ok=True)
            
            dice_loss_list = []

            

            for _, val_data in enumerate(val_loader):

                path = val_set.ori_path[val_data['Index']]
                image_name = os.path.basename(path)

                idx += 1

                with torch.no_grad():

                    segnet.set_input(val_data)
                    segnet.validate()
                    val_errors = segnet.get_current_errors_val()
                    loss_dice_val = val_errors['loss_dice_val']
                    dice_loss_list.append(loss_dice_val)
                    print(1 - loss_dice_val)
                    val_viz = segnet.get_current_visuals_val()

                soft_fake_img = val_viz['soft_pred_val']
                fake_img = val_viz['pred_val']
                fake_img_int = (fake_img * 255.0).round()
                ori_img = val_viz['img_seen_val']
                ori_img_int = (ori_img * 255.0).round()
                tg_img = val_viz['gth_val']
                tg_img_int = (tg_img * 255.0).round()

                visual_img = np.concatenate((ori_img_int[0,0,:,:],
                                      tg_img_int[0,0,:,:],
                                      fake_img_int[0,0,:,:]), axis=1)

                Metrics.save_img(visual_img, '{}/{}_fake.png'.format(result_dir, image_name))
                

                Metrics.save_img(visual_img, f'{result_dir}/{image_name}_lowdice_{1 - loss_dice_val:.3f}.png')

                # produce final estimation using a softmax layer.
                exp_x = np.exp(soft_fake_img)
                soft_fake_results = exp_x / np.sum(exp_x,axis=1,keepdims=True)

                # generation
                np.save('{}/{}_softfake.npy'.format(result_dir, image_name), soft_fake_results)
                np.save('{}/{}_fake.npy'.format(result_dir, image_name), fake_img)
                np.save('{}/{}_ori.npy'.format(result_dir, image_name), ori_img)
                np.save('{}/{}_tg.npy'.format(result_dir, image_name), tg_img)
                


                logger.info('End Model Evaluation.')
            
                


