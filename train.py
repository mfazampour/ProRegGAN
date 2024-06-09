import time
import datetime
import os

import torch
import numpy as np
import csv

from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import SimpleITK as sitk
os.environ["TORCHIO_HIDE_CITATION_PROMPT"] = str(1)
import torchio
import wandb

from options.train_options import TrainOptions
from data import create_dataset, CustomDatasetDataLoader
from models import create_model, BaseModel
from util.visualizer import Visualizer
from models.multitask_parent import Multitask

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    parser = TrainOptions()
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

    dataset, dataset_val, dataset_test, model, optimize_time, total_iters, visualizer, writer = initialize_experiment(opt)

    train(dataset, dataset_val, dataset_test, model, opt, optimize_time, total_iters, visualizer, writer)


def train(dataset, dataset_val, dataset_test, model, opt, optimize_time, total_iters, visualizer, writer):
    model.init_losses()
    for epoch in range(opt.epoch_count,
                       opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        dataset.set_epoch(epoch)
        total_iters = train_one_epoch(dataset, dataset_val, epoch, epoch_iter, iter_data_time, model, opt,
                                      optimize_time, total_iters, visualizer, writer)
        evaluate_model(dataset_test, model, total_iters, writer, 10, opt.save_volume,  # only run for 10 test images
                       opt.checkpoints_dir, opt.evaluation_freq, epoch, save_pdf=True)
        wandb.save(os.path.join(writer.log_dir, 'pdf/', f'{epoch:03}', "*.pdf"), os.path.join(writer.log_dir, 'pdf/'))

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate(epoch=epoch)  # update learning rates at the end of every epoch.


def train_one_epoch(dataset, dataset_val, epoch, epoch_iter, iter_data_time, model, opt, optimize_time, total_iters,
                    visualizer, writer):
    losses_total = []
    for i, data in enumerate(dataset):  # inner loop within one epoch

        iter_start_time = time.time()  # timer for computation per iteration
        # if total_iters % opt.print_freq == 0:
        t_data = iter_start_time - iter_data_time

        # check the number of open figures
        if len(plt.get_fignums()) > 100:
            plt.close('all')

        total_iters += opt.batch_size
        epoch_iter += opt.batch_size
        if len(opt.gpu_ids) > 0:
            torch.cuda.synchronize()
        optimize_start_time = time.time()
        if epoch == opt.epoch_count and i == 0:
            init_model(data, model, opt)

        iter_data_time = time.time()

        # evaluate model performance every evaluation_freq iterations
        if int(total_iters / opt.batch_size) % opt.evaluation_freq == 1:
            evaluate_model(dataset_val, model, total_iters, writer, opt.num_validation_samples, opt.save_volume,
                           opt.checkpoints_dir, opt.evaluation_freq, epoch)

        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

        if len(opt.gpu_ids) > 0:
            torch.cuda.synchronize()

        new_time = (time.time() - optimize_start_time) / opt.batch_size
        optimize_time = new_time * 0.5 if optimize_time == -1 else new_time * 0.02 + 0.98 * optimize_time

        losses_total.append(model.get_current_losses())
        log_training(data, epoch, epoch_iter, losses_total, model, opt, optimize_time, t_data, total_iters, visualizer,
                     writer)

        # cache our latest model every <save_latest_freq> iterations
        save_model(epoch, model, opt, total_iters)
    return total_iters


def init_model(data, model, opt):
    print("initializing data dependent parts of the network")
    model.data_dependent_initialize(data)
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    model.parallelize()


def save_model(epoch, model, opt, total_iters):
    if total_iters % opt.save_latest_freq == 0:
        print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
        print(opt.name)  # it's useful to occasionally show the experiment name on console
        save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
        model.save_networks(save_suffix)


def log_training(data, epoch, epoch_iter, losses_total, model, opt, optimize_time, t_data, total_iters, visualizer,
                 writer):
    if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
        losses = model.get_current_losses()
        visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)

    # display images on visdom and save images to a HTML file
    if total_iters % opt.display_freq == 0:
        display_results(data, epoch, model, opt, total_iters, visualizer, writer)
        if len(losses_total) > 0:
            loss_aggregate = {}
            for key in losses_total[0].keys():
                loss_aggregate[key] = np.nanmean([losses.get(key, np.NaN) for losses in losses_total])
            for key in loss_aggregate:
                writer.add_scalar(f'train-losses/{key}', scalar_value=loss_aggregate[key], global_step=total_iters)
        losses_total.clear()


def initialize_experiment(opt):
    dataset = create_dataset(opt, mode='train')  # create a dataset given opt.dataset_mode and other options
    dataset_val = create_dataset(opt, mode='validation')  # validation dataset
    dataset_test = create_dataset(opt, mode='test')
    dataset_size = len(dataset)  # get the number of images in the dataset.
    model = create_model(opt)  # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0  # the total number of training iterations
    print('visualizer started')
    optimize_time = -1
    opt.tensorboard_path = os.path.join(opt.checkpoints_dir, opt.name + '/',
                                        datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    os.makedirs(opt.tensorboard_path, exist_ok=True)
    writer = SummaryWriter(opt.tensorboard_path)
    print('tensorboard started')
    # start wandb
    wandb.login(key=opt.wandb_key)
    wandb.init(project='prostate', config=opt, name=opt.name)
    return dataset, dataset_val, dataset_test, model, optimize_time, total_iters, visualizer, writer


def display_results(data, epoch, model: BaseModel, opt, total_iters, visualizer, writer):
    losses = model.get_current_losses()  # read losses before setting to no_grad for validation
    data = data
    model.eval()  # change networks to eval mode
    with torch.no_grad():
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
    save_result = total_iters % opt.update_html_freq == 0
    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
    if isinstance(model, Multitask):
        model.compute_landmark_loss()
    model.train()  # change networks back to train mode
    wandb_dict = model.log_tensorboard(writer, losses, total_iters, save_gif=False, mode='train', epoch=epoch)
    if isinstance(model, Multitask):
        mt_dict, tag = model.log_mt_tensorboard(model.real_A, model.real_B, model.moving, writer, global_step=total_iters, mode='train')
        if tag is not None:
            wandb_dict[tag].update(mt_dict[tag])

    wandb.log(wandb_dict)


def evaluate_model(dataset: CustomDatasetDataLoader, model: BaseModel, total_iters, writer, num_validation_samples, save_volume: bool,
                   checkpoint_path: str, evaluation_freq, epoch, save_pdf=False):
    print('evaluating model on labeled data')
    losses_total = []
    keys = []
    loss_aggregate = {}
    land_rig = []
    land_def = []
    land_beg = []
    wandb_dict = {}
    for j, (val_data) in enumerate(dataset):
        model.eval()  # change networks to eval mode
        if j > num_validation_samples:
            break
        stride = np.max([int(evaluation_freq/num_validation_samples), 1])
        with torch.no_grad():
            model.set_input(val_data)  # unpack data from data loader
            model.forward()  # run inference
            model.calculate_loss_values()  # get the loss values
            model.compute_visuals()
            losses = model.get_current_losses()
            losses_total.append(losses)
            model.get_current_visuals()
            subject_dict = model.log_tensorboard(writer=writer, losses=None, global_step=total_iters + j * stride,
                                             save_gif=False, use_image_name=True, mode=f'{dataset.dataset.mode}-',
                                             epoch=epoch, save_pdf=save_pdf)
            if isinstance(model, Multitask):
                tmp_dict, tag = model.log_mt_tensorboard(model.real_A, model.real_B, model.moving, writer=writer,
                                         global_step=total_iters + j * stride, use_image_name=True,
                                         mode=f'{dataset.dataset.mode}-')
                if tag is not None:
                    subject_dict[tag].update(tmp_dict[tag])
            if save_volume and isinstance(model, Multitask):
                os.makedirs(os.path.join(checkpoint_path, 'vol'), exist_ok=True)
                img = torchio.ScalarImage(tensor=model.moving[0, ...].detach().cpu())
                img.save(os.path.join(checkpoint_path, f'vol/{model.patient[0]}_fake.nii'))

            wandb_dict.update(subject_dict)
        keys = losses.keys()
    for key in keys:
        loss_aggregate[key] = np.nanmean([losses.get(key, np.NaN) for losses in losses_total])
    for key in loss_aggregate:
        writer.add_scalar(f'{dataset.dataset.mode}-losses/{key}', scalar_value=loss_aggregate[key], global_step=total_iters)
    wandb_dict[f'{dataset.dataset.mode}-losses'] = loss_aggregate
    wandb.log(wandb_dict)


if __name__ == '__main__':
    main()