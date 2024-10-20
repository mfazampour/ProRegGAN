"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from datetime import datetime

import wandb

from options.test_options import TestOptions
from util.visualizer import save_images
import torch
import numpy as np
import csv
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from torch.utils.tensorboard import SummaryWriter
from models.multitask_parent import Multitask
from models.base_model import BaseModel


def main():
    parser = TestOptions()
    opt = parser.parse()  # get training options

    try:
        from polyaxon_helper import (
            get_outputs_path,
            get_data_paths,
        )

        base_path = get_data_paths()
        print("You are running on the cluster :)")
        opt.dataroot = base_path['data1'] + opt.dataroot
        opt.checkpoints_dir = get_outputs_path()
        opt.display_id = -1  # no visdom available on the cluster
        parser.print_options(opt)
    except Exception as e:
        print(e)
        print("You are Running on the local Machine")

    dataset_test = create_dataset(opt, mode='test')
    # dataset_val = create_dataset(opt, mode='validation')  # validation dataset

    dataset_size = len(dataset_test)  # get the number of images in the dataset.

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    if not isinstance(model, Multitask) and not isinstance(model, BaseModel):
        # raise an error if the model is not a multitask model, test.py is only for multitask models
        raise ValueError('This script is only for multitask models')

    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0  # the total number of training iterations
    print('visualizer started')
    optimize_time = -1
    opt.tensorboard_path = os.path.join(opt.checkpoints_dir, opt.name + '/',
                                        datetime.now().strftime('%Y-%m-%d_%H-%M'))
    os.makedirs(opt.tensorboard_path, exist_ok=True)
    writer = SummaryWriter(opt.tensorboard_path)
    print('tensorboard started')

    # Start wandb
    wandb.login(key=opt.wandb_key)
    wandb.init(project='prostate', config=vars(opt), name=opt.name)

    model.eval()

    print('evaluating model on labeled data')
    losses_total = []
    keys = []
    loss_aggregate = {}
    land_def = []
    land_beg = []
    dice_dif = []
    dice_move = []
    dice_warp = []

    for j, (test_data) in enumerate(dataset_test):
        model.eval()  # change networks to eval mode
        with torch.no_grad():
            model.set_input(test_data)  # unpack data from data loader
            model.forward()  # run inference
            model.calculate_loss_values()  # get the loss values
            model.compute_visuals()
            losses = model.get_current_losses()
            losses_total.append(losses)
            model.get_current_visuals()
            landmarks_beg, landmarks_def = model.get_current_landmark_distances()
            land_beg.append(landmarks_beg.item())
            land_def.append(landmarks_def.item())
            dice_dif.append(model.loss_diff_dice)
            dice_move.append(model.loss_moving_dice)
            dice_warp.append(model.loss_warped_dice)
            model.log_tensorboard(writer, None, j, save_gif=False, use_image_name=True, mode='test', epoch=0)

            # Log to wandb
            wandb.log({
                'test/landmarks_beg': landmarks_beg.item(),
                'test/landmarks_def': landmarks_def.item(),
                'test/diff_dice': model.loss_diff_dice,
                'test/moving_dice': model.loss_moving_dice,
                'test/warped_dice': model.loss_warped_dice,
                **{f'test/losses/{key}': losses[key] for key in losses}
            })

        keys = losses.keys()

    for key in keys:
        loss_aggregate[key] = np.mean([losses[key] for losses in losses_total])
    for key in loss_aggregate:
        writer.add_scalar(f'avg-losses/{key}', scalar_value=loss_aggregate[key], global_step=1)
        # Log aggregated loss to wandb
        wandb.log({f'avg-losses/{key}': loss_aggregate[key]})

    # Finish wandb run
    wandb.finish()


if __name__ == '__main__':
    main()

