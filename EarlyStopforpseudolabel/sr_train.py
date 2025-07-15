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
    parser.add_argument('-c', '--config', type=str, default='config/SR3_EffNet.json',
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
    
    
    # ## ===============================================================================
    # # Segmentation Network Definition.
    # # model
    segnet = create_forward(opt['segnet'])
    logger.info('Initial SegNet Finished')
    # ## ===============================================================================
    
    current_step = 0
    current_epoch = 0
    n_epoch = 12


    while current_epoch < n_epoch:
        current_epoch += 1
        for _, train_data in enumerate(train_loader):
            current_step += 1

            ## run a training step
            segnet.set_input(train_data)
            segnet.optimize_parameters()

    # =======================================


            if current_step % opt['train']['print_freq'] == 0:
                tr_error = segnet.get_current_errors_tr()
                print(tr_error)
                segnet.track_scalar_in_tb(tb_logger, tr_error, current_step)


        # validation
        if current_epoch % opt['train']['val_freq'] == 0:
            logger.info('epoch: {}, iter: {}, begin validation'.format(
                current_epoch, current_step))
            avg_val_errors = 0.0
            idx = 0
            result_dir = opt['path']['results']
            os.makedirs(result_dir, exist_ok=True)

            visual_img = []

            for _,  val_data in enumerate(val_loader):
                idx += 1

                with torch.no_grad():

                    segnet.set_input(val_data)
                    segnet.validate()
                    val_errors = segnet.get_current_errors_val()
                    val_viz = segnet.get_current_visuals_val()

                fake_img_int = (val_viz['pred_val'] * 255.0).round()
                ori_img_int = ((val_viz['img_seen_val'] + 12.0) * 10.0).round()
                tg_img_int = (val_viz['gth_val'] * 255.0).round()

                avg_val_errors += val_errors['loss_dice_val']


                tmp = np.concatenate((ori_img_int[0,0,:,:],
                                      tg_img_int[0,0,:,:],
                                      fake_img_int[0,0,:,:]), axis=1)

                if idx == 1:
                    visual_img = tmp
                else:
                    visual_img = np.concatenate((visual_img, tmp), axis = 0)

                if wandb_logger:
                    wandb_logger.log_image(f'validation_{idx}', visual_img)

            # generation
            Metrics.save_img(visual_img, '{}/epoch{:0>3d}_{}.png'.format(result_dir, current_epoch, idx))

            avg_val_errors = avg_val_errors / idx
            # log
            logger.info('# Validation # ERROR: {:.4e}'.format(avg_val_errors))
            logger_val = logging.getLogger('val')  # validation logger
            logger_val.info('<epoch:{:3d}, iter:{:8,d}> error: {:.4e}'.format(
                current_epoch, current_step, avg_val_errors))
            # tensorboard logger
            tb_logger.add_scalar('error', avg_val_errors, current_step)
            logger.info('validation completed')

            if wandb_logger:
                wandb_logger.log_metrics({
                    'validation/val_psnr': avg_val_errors,
                    'validation/val_step': val_step
                })
                val_step += 1

        if current_epoch % opt['train']['save_checkpoint_freq'] == 0:
            logger.info('Saving models and training states.')
            # diffusion.save_network(current_epoch, current_step)
            segnet.save(opt['path']['checkpoint'], 'latest')
            segnet.save(opt['path']['checkpoint'], current_epoch)

            if wandb_logger and opt['log_wandb_ckpt']:
                wandb_logger.log_checkpoint(current_epoch, current_step)

        if wandb_logger:
            wandb_logger.log_metrics({'epoch': current_epoch - 1})

        segnet.update_learning_rate()

    # save model
    logger.info('End of training.')
    
    
