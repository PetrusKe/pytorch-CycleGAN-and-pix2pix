"""Model class template

This module provides a template for users to implement custom models.
You can specify '--model template' to use this model.
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.
Given input-output pairs (data_A, data_B), it learns a network netG that can minimize the following L1 loss:
    min_<netG> ||netG(data_A) - data_B||_1
You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import cv2

class FlawDetectorModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(norm='batch', netG='unet_256', netD='fd', dataset_mode='aligned')  # You can rewrite default values for this model. For example, this model usually uses aligned dataset as its dataset.
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['G_L1', 'G_Adv', 'D_detect']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'pred_D', 'gt_D']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks to save and load networks.
        # you can use opt.isTrain to specify different behaviors for training and test. For example, some networks will not be used during test, and you don't need to load them.
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
    
        if self.isTrain:  # only defined during training time
            # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
            # We also provide a GANLoss class "networks.GANLoss". self.criterionGAN = networks.GANLoss().to(self.device)
            
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionAdv = networks.MinimumFDLoss()
            self.criterionD = networks.FlawDetectorCriterion()

            # define and initialize optimizers. You can define one optimizer for each network.
            # If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=0.0025, weight_decay=0.0005)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        
    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

    def backward_D(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        self.pred_D = self.netD(fake_AB.detach())

        self.gt_D = self.gt_D_func()
        self.loss_D_detect = self.criterionD(self.pred_D, self.gt_D)
        self.loss_D = self.loss_D_detect
        self.loss_D.backward()
    
    def gt_D_func(self):
        diff = torch.abs(self.fake_B - self.real_B)

        clip_threshold = 30     # between [0, 255]

        blur_kernel = int(diff.shape[-1] / 8.0)
        blur_kernel = blur_kernel + 1 if blur_kernel % 2 == 0 else blur_kernel

        reblur_kernel = int(diff.shape[-1] / 4.0)
        reblur_kernel = reblur_kernel + 1 if reblur_kernel % 2 == 0 else reblur_kernel

        gtD = diff.data.cpu().numpy() * 255.0 / 3.0
        gtD = gtD.sum(1, keepdims=True)

        for sdx in range(0, gtD.shape[0]):
            s_diff = gtD[sdx, 0, ...]

            s_diff_mean = np.mean(s_diff)
            s_diff -= s_diff_mean
            s_diff[s_diff < 0] = 0

            # for i in range(0, 1):
                # s_diff_mean = np.sum(s_diff) / np.count_nonzero(s_diff)
                # s_diff[s_diff < s_diff_mean] = 0

            # blurred = cv2.medianBlur(s_diff, 3)
            # blurred = cv2.threshold(blurred, clip_threshold, 255, cv2.THRESH_TOZERO)[1]
            blurred = s_diff

            # for i in range(0, 1):
                # blurred = cv2.dilate(blurred, None, iterations=1)
                # blurred = cv2.GaussianBlur(blurred, (blur_kernel, blur_kernel), 0)

                # blurred = cv2.blur(diff, (blur_kernel, blur_kernel))
                # blurred =  (blurred - blurred.min()) / (blurred.max() - blurred.min() + 1e-6)

            # dilated = cv2.dilate(blurred, None, iterations=2)
            # eroded = cv2.erode(blurred, None, iterations=1)
            # reblurred = eroded
            reblurred = blurred
            # clipped = cv2.threshold(blurred, clip_threshold, 255, cv2.THRESH_TOZERO)[1]

            # # eroded = cv2.erode(clipped, None, iterations=1)
            # dilated = cv2.dilate(clipped, None, iterations=2)

            # reblurred = cv2.GaussianBlur(dilated, (reblur_kernel, reblur_kernel), 0)
            finished = (reblurred - reblurred.min()) / (reblurred.max() - reblurred.min() + 1e-6)
            gtD[sdx, 0, ...] = finished

        return torch.FloatTensor(gtD).cuda()

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_Adv = self.criterionAdv(pred_fake) * 0.001

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)  * self.opt.lambda_L1

        self.loss_G = self.loss_G_Adv + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()                   # compute fake images: G(A)
        # update D
        
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights

        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
