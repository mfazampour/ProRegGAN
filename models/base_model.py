import inspect
import os
from inspect import getmembers, isfunction
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod

from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from models import networks
from models.multitask_parent import Multitask


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        if opt.preprocess != 'scale_width':  # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.loss_functions = []
        self.metric = 0  # used for learning rate policy 'plateau'
        self.min_Ganloss = 1000.0
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.patient = ['']


    @staticmethod
    def dict_grad_hook_factory(add_func=lambda x: x):
        saved_dict = dict()

        def hook_gen(name):
            def grad_hook(grad):
                saved_vals = add_func(grad)
                saved_dict[name] = saved_vals
            return grad_hook
        return hook_gen, saved_dict

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = opt.epoch
            self.load_networks(load_suffix)

        self.print_networks(opt.verbose)

    def parallelize(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                setattr(self, 'net' + name, torch.nn.DataParallel(net, self.opt.gpu_ids))

    def data_dependent_initialize(self, data):
        pass

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                if name != "G" and name != "D":
                    net.train()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """ Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self, epoch=0):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]['lr']
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate %.7f -> %.7f' % (old_lr, lr))

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):                visual_ret[name] = getattr(self, name)
        return visual_ret

    def init_losses(self):
        """Init losses to None"""
        for name in self.loss_names:
            if isinstance(name, str):
                setattr(self, 'loss_' + name, None)

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                if getattr(self, 'loss_' + name) is not None:
                    errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_smallest_network(self):
        for name in self.model_names:
            current_loss = self.get_current_losses()['G']
            if current_loss < self.min_Ganloss:
                save_filename_min = "smallest_loss_model"
                save_path_min = os.path.join(self.save_dir, save_filename_min)
                net_min = getattr(self, 'net' + name)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():

                    torch.save(net_min.module.cpu().state_dict(), save_path_min)
                    net_min.cuda(self.gpu_ids[0])
                else:
                    torch.save(net_min.cpu().state_dict(), save_path_min)
                self.min_Ganloss = current_loss



    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])

                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:

            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)

                if name in Multitask.get_model_names() and self.opt.load_init_multitask_models is not None:
                    base_path = self.opt.load_init_multitask_models
                else:
                    base_path = self.opt.load_init_models
                if self.opt.isTrain and self.opt.pretrained_name is not None:
                    join_ = os.path.join(base_path, self.opt.pretrained_name)
                    load_dir = join_
                else:
                    load_dir = base_path
                load_path = os.path.join(load_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                if not os.path.isfile(load_path):
                    print(f'failed to load {name} network. {load_path} does not exits')
                    continue
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                #    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))

                print(f"name {name}")
                pre_params = state_dict
                new_params = net.state_dict()

                for key in new_params:
                    if key in pre_params:
                        if new_params[key].shape == pre_params[key].shape:  # replace the values when sizes match
                            new_params[key] = pre_params[key]
                        else:
                            print(f'shape does not match for {key} in model {name}, {new_params[key].shape} vs {pre_params[key].shape}')
                    else:
                        print(f'{key} from new model does not exist in model {name}')

                net.load_state_dict(new_params)

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def generate_visuals_for_evaluation(self, data, mode):
        return {}


    def calculate_loss_values(self):
        """Can generate the loss values without calling optimize. Use it only after calling forward().
        Created only for testing
        Returns
        -------

        """
        for fun in self.loss_functions:
            getattr(self, fun)()

    def log_tensorboard(self, writer: SummaryWriter, losses: OrderedDict = None, global_step: int = 0,
                        save_gif=False, use_image_name=False, mode='', epoch: int = 0, save_pdf=False):
        fig = self.create_figure(writer=writer, global_step=global_step, save_gif=save_gif)
        fig.suptitle(f'ID {self.patient[0]}')
        tag = f'{self.patient[0]}' if use_image_name else ''
        writer.add_figure(tag=mode + tag + '/GAN', figure=fig, global_step=global_step, close=False)
        # fig.clf()
        # plt.close(fig)

        # save fig as pdf
        if save_pdf:
            pdf_dir = os.path.join(writer.log_dir, 'pdf/', f'{epoch:03}')
            if not os.path.isdir(pdf_dir):
                os.makedirs(pdf_dir, exist_ok=True)
            fig.savefig(f'{os.path.join(pdf_dir, tag + "GAN")}.pdf')

        wandb_dict = {mode + tag: {'GAN': fig}}
        self.log_loss_tensorboard(global_step, losses, wandb_dict, writer)
        return wandb_dict

    def create_figure(self, writer: SummaryWriter, global_step: int = 0, save_gif=False) -> plt.Figure:
        raise NotImplementedError()

    def log_loss_tensorboard(self, global_step, losses, wandb_dict, writer):
        if losses is not None:
            for key in losses:
                writer.add_scalar(f'losses/{key}', scalar_value=losses[key], global_step=global_step)
                wandb_dict['loss'] = losses
