import torch
import torch.nn as nn
import itertools
from .base_model import BaseModel
from . import networks
from . import vgg
import torch.nn.functional as F
import numpy as np
import skimage.measure as measure
import code
import torchvision.transforms as transforms
from torch.autograd import Variable
from pytorch_ssim import SSIM
import clip
from CLIP.clip import load
from collections import OrderedDict


def gauss_kernel(size=5, device=torch.device('cpu'), channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel


def downsample(x):
    return x[:, :, ::2, ::2]


def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out


def guass_pyramid(img, kernel):
    filtered = conv_gauss(img, kernel)
    down = downsample(filtered)
    return down


def preprocess_feature(img, model):
    clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    img_resize = transforms.Resize((224,224))
    img=img_resize(img)
    img=clip_normalizer(img)
    image_features = model.encode_image(img)
    image_features_norm = image_features / image_features.clone().norm(dim=-1, keepdim=True)
    return image_features_norm


class IRLRModel(BaseModel, torch.nn.Module):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=False)  # default CycleGAN did not use dropout
        parser.add_argument('--blurKernel', type=int, default=5, help='maximum R for gaussian kernel')
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')

        parser.add_argument('--lambda_DisNCE', type=float, default=1.0, help='weight for Dis NCE loss')
        parser.add_argument('--lambda_MSE', type=float, default=1.0, help='weight for MSE loss')
        parser.add_argument('--lambda_L1', type=float, default=0.1, help='weight for MSE loss')

        parser.add_argument('--adv_nce_layers', type=str, default='0,3,7,11', help='compute NCE loss on which layers')
        parser.add_argument('--gen_nce_layers', type=str, default='0,2,4,8,12', help='compute NCE loss on which layers')
        parser.add_argument('--netFGen', type=str, default='mlp_sample',
                            choices=['mlp_sample', 'non_localOne'])
        parser.add_argument('--netFAdvRain', type=str, default='non_localOne',
                            choices=['mlp_sample', 'non_localOne'])
        parser.add_argument('--netFAdvBack', type=str, default='non_localOne',
                            choices=['mlp_sample', 'non_localOne'])
        parser.add_argument('--netF_nc', type=int, default=128)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches_pos', type=int, default=8, help='number of patches per layer')
        parser.add_argument('--num_patches_neg', type=int, default=128, help='number of patches per layer')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0,serial_batches=False)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        torch.nn.Module.__init__(self)
        self.loss_names = ['idt_T', 'ssim', 'R', 'res', 'MP', 'comp', 'T', 'CLIP', 'CLIP_MSE', 'margin']

        if self.isTrain:
            self.visual_names = ['fake_Ts', 'real_T', 'fake_Rs', 'real_R']
        else:
            self.visual_names = ['fake_Ts', 'real_T', 'real_I', 'fake_Rs']

        if self.isTrain:
            self.model_names = ['G_T', 'Learn_Prompt']
        else:  # during test time, only load Gs
            self.model_names = ['G_T']

        self.vgg = vgg.Vgg19(requires_grad=False).to(self.device)
        # Define generator of synthesis net
        self.netG_T = networks.define_G(opt.input_nc, opt.input_nc, opt.ngf, opt.netG, self.device, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.clip_model, _ = clip.load("ViT-B/32", device=self.device,
                                           download_root="./pretrained_models/clip_model/")
            self.set_requires_grad([self.clip_model], False)
            # self.netLearn_Prompt = networks.Prompts(self.clip_model,
            #                                         [" ".join(["X"]*(16)), " ".join(["X"]*(16)),
            #                                          " ".join(["X"]*(16))]).cuda()
            self.netLearn_Prompt = networks.Prompts(self.clip_model, './pretrained_models/init_pretrained_models/best_prompt_round0.pth').cuda()
            self.netLearn_Prompt = torch.nn.DataParallel(self.netLearn_Prompt, self.gpu_ids)
            if not opt.continue_train:
                load_path = './pretrained_models/init_pretrained_models/model.pth'
                print('loading the model from %s' % load_path)
                state_dict = torch.load(load_path, map_location=str(self.device))
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = 'module.' + k  # remove `module.`
                    new_state_dict[name] = v
                self.netG_T.load_state_dict(new_state_dict)
            # torch.nn.utils.clip_grad_norm_(self.netG_T.parameters(), 0.25)
            # torch.nn.utils.clip_grad_norm_(self.netG_R.parameters(), 0.25)
            self.criterionGradient = torch.nn.L1Loss()
            self.criterionSSIM = SSIM(window_size=11).to(self.device)
            self.criterionVgg = networks.VGGLoss1(self.device, vgg=self.vgg, normalize=False)
            self.criterionCLIP = networks.L_clip_from_feature().to(self.device)
            self.criterionCLIP_MSE = networks.L_clip_MSE().to(self.device)
            self.criterionCLIP_R = networks.L_clip_from_feature_R().to(self.device)
            self.criterionL_margin = networks.four_margin_loss(0.9, 0.2)
            self.criterionPromptNCE = networks.L_prompt_NCE().to(self.device)

            self.res_model, _ = load("RN50", device=self.device, download_root="./pretrained_models/clip_model/")
            self.set_requires_grad([self.res_model], False)

            self.text_encoder = networks.TextEncoder(self.clip_model)

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_T.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_prompt = torch.optim.Adam(itertools.chain(self.netLearn_Prompt.parameters()),
                                                     lr=opt.lr * 10, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_prompt)

        self.criterionIdt = torch.nn.MSELoss()
        self.criterionL1 = torch.nn.L1Loss()

        resSize = 64
        self.k_sz = np.linspace(opt.batch_size, self.opt.blurKernel, 80)  # for synthetic images

        self.fake_T = torch.zeros(self.opt.batch_size, 3, 256, 256).to(self.device)
        self.fake_Ts = [self.fake_T]
        self.fake_R = torch.zeros(self.opt.batch_size, 3, 256, 256).to(self.device)
        self.fake_Rs = [self.fake_R]

        # Pass invalid data
        self.trainFlag = True

        ''' We use both real-world data and synthetic data. If 'self.isNatural' is True, the data loaded is real-world
        image paris. Otherwise, we use 'self.syn' to synthesize data.'''
        self.isNatural = False
        self.syn = networks.SynData(self.device)
        # self.load_curv = networks.CurvMap()
        self.real_I = None
        self.real_I2 = None
        self.real_I4 = None
        self.real_T = None
        self.real_T2 = None
        self.real_T4 = None
        self.real_R = None
        self.real_R2 = None
        self.real_R4 = None
        self.alpha = None
        self.kernel = gauss_kernel(size=5, device=self.device, channels=3)
        self.isNatural = None

    def set_input(self, input):
        """Unpack input data from the dataloader, perform necessary pre-processing steps and synthesize data.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.alpha = []
        T_batch = input['T']
        I_batch = input['I']
        R_batch = torch.zeros_like(T_batch)
        self.isNatural = input['isNatural']
        with torch.no_grad():
            if self.isTrain:
                for i in range(T_batch.shape[0]):
                    if input['isNatural'][i] == 1:
                        isNatural_cur = True
                    else:
                        isNatural_cur = False
                    # self.real_T2 = input['T2'].to(self.device)
                    # self.real_T4 = input['T4'].to(self.device)
                    if not isNatural_cur:  # Skip these procedures, if the data is from real-world.
                        T = input['T'][i].unsqueeze(0).to(self.device)
                        R = input['R'][i].unsqueeze(0).to(self.device)
                        # if torch.mean(T) * 1 / 2 > torch.mean(R):
                        #     self.trainFlag = False
                        #     return
                        _, R, I, alpha = self.syn(T, R, self.k_sz)  # Synthesize data
                        self.alpha.append(round(alpha, 1))
                        # if T.max() < 0.15 or R.max() < 0.15 or I.max() < 0.1:
                        #     self.trainFlag = False
                        #     return
                        self.real_R = R.float().to(self.device)
                        self.real_R2 = guass_pyramid(self.real_R, self.kernel)
                        self.real_R4 = guass_pyramid(self.real_R2, self.kernel)
                        T_batch[i] = T[0]
                        I_batch[i] = I[0]
                        R_batch[i] = R[0].float()
                    else:
                        I_batch[i] = input['I'][i]
                        T_batch[i] = input['T'][i]
                        R_batch[i] = input['R'][i]
                        self.alpha.append(0.)
                self.real_R = R_batch.to(self.device)
                self.real_R2 = guass_pyramid(self.real_R, self.kernel)
                self.real_R4 = guass_pyramid(self.real_R2, self.kernel)
            else:  # Test
                self.image_paths = input['B_paths']
                I_batch = input['I']
                T_batch = input['T']

        self.real_T = T_batch.to(self.device)
        self.real_T2 = guass_pyramid(self.real_T, self.kernel)
        self.real_T4 = guass_pyramid(self.real_T2, self.kernel)
        self.real_I = I_batch.to(self.device)
        self.real_I2 = guass_pyramid(self.real_I, self.kernel)
        self.real_I4 = guass_pyramid(self.real_I2, self.kernel)

    def get_c(self):
        b, c, w, h = self.real_I.shape
        return torch.zeros((b, self.opt.ngf * 4, w//4, h//4))

    def init(self):
        b, c, h, w = self.real_I.shape
        self.h = Variable(torch.zeros(b, 64, h, w, device=self.device))
        self.c = Variable(torch.zeros(b, 64, h, w, device=self.device))
        self.fake_T = self.real_I.clone().detach()
        self.fake_Ts = [self.fake_T]
        self.fake_R = self.real_I.clone().detach()
        self.fake_Rs = [self.fake_R]

        self.rcmaps = []

    def forward(self):
        self.init()
        for i in range(3):
            self.h, self.c, self.c_map, self.fake_R, self.fake_T4, self.fake_T2, self.fake_T = \
                self.netG_T(self.real_I, self.fake_Ts[-1], self.h, self.c)

            self.rcmaps.append(self.c_map)
            self.fake_Rs.append(self.fake_R)
            self.fake_Ts.append(self.fake_T)

        # clip operation in test
        if not self.isTrain:
            for i in range(len(self.fake_Ts)):
                self.fake_Ts[i] = torch.clamp(self.fake_Ts[i], min=0, max=1)
            for i in range(len(self.fake_Rs)):
                self.fake_Rs[i] = torch.clamp(self.fake_Rs[i], min=0, max=1)

    def get_text_feats(self):
        # state_dict = torch.load('./pretrained_models/init_pretrained_models/best_prompt_round0.pth')
        # # create new OrderedDict that does not contain `module.`
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = k[7:]  # remove `module.`
        #     new_state_dict[name] = v
        # embedding_prompt = nn.Parameter(new_state_dict['embedding_prompt']).cuda()
        embedding_prompt = self.netLearn_Prompt.module.embedding_prompt
        embedding_prompt.requires_grad = False
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in [" ".join(["X"] * 16)]])
        self.text_features = self.text_encoder(embedding_prompt, tokenized_prompts)

    # def backward_D_basic(self, netD, real, fake):
    #     """Calculate GAN loss for the discriminator
    #
    #     Parameters:
    #         netD (network)      -- the discriminator D
    #         real (tensor array) -- real images
    #         fake (tensor array) -- images generated by a generator
    #
    #     Return the discriminator loss.
    #     We also call loss_D.backward() to calculate the gradients.
    #     """
    #     # Real
    #     pred_real = netD(real)
    #     loss_D_real = self.criterionGAN(pred_real, True)
    #     # Fake
    #     pred_fake = netD(fake.detach())
    #     loss_D_fake = self.criterionGAN(pred_fake, False)
    #     # Combined loss and calculate gradients
    #     loss_D = (loss_D_real + loss_D_fake) * 0.5
    #     loss_D.backward()
    #     return loss_D
    #
    # def backward_D(self):
    #     """Calculate GAN loss for discriminator D_syn"""
    #     self.loss_D_syn = self.backward_D_basic(self.netD, self.real_T, self.fake_T)

    def backward_prompt(self):
        self.loss_prompt = 0.
        self.loss_promptNCE = 0.
        self.loss_margin = 0.
        real_T_feature = preprocess_feature(self.real_T, self.clip_model)
        fake_T_feature = preprocess_feature(self.fake_T.detach(), self.clip_model)
        real_I_feature = preprocess_feature(self.real_I, self.clip_model)
        labels = torch.ones(real_T_feature.shape[0]).long().to(self.device)
        self.loss_margin = self.criterionL_margin(self.netLearn_Prompt(real_I_feature.cuda()),
                                                  self.netLearn_Prompt(real_T_feature.cuda()), labels, 3,
                                                  self.netLearn_Prompt(fake_T_feature.cuda()))
        self.loss_prompt = self.loss_margin
        self.loss_prompt.backward()

    def backward_G(self):
        self.loss_idt_T = 0.0          # L_pixel on T
        self.loss_idt_R = 0.0          # L_pixel on R
        self.loss_res = 0.0            # L_residual: residual reconstruction loss
        self.loss_MP = 0.0             # L_MP: multi-scale perceptual loss
        self.loss_R = 0.0
        self.loss_CLIP = 0.0
        self.loss_CLIP_MSE = 0.0
        self.loss_CLIP_R = 0.0
        self.loss_CLIP_MSE_R = 0.0
        self.loss_img_cont = 0.0
        self.loss_comp = 0.0
        self.loss_ssim = 0.0
        self.loss_mix = 0.0
        self.loss_cont_B = 0.0
        self.loss_cont_R = 0.0

        self.loss_prompt = 0.
        self.loss_promptNCE = 0.
        self.loss_margin = 0.

        iter_num = len(self.fake_Ts)

        sigma = 0.85

        for i in range(iter_num):
            if i > 0:
                # self.loss_idt_T += self.criterionIdt(self.fake_Ts[i], self.real_T) * np.power(sigma, iter_num - i - 1)
                self.loss_idt_T += self.criterionL1(self.fake_Ts[i], self.real_T) * np.power(sigma, iter_num - i - 1)
                self.loss_ssim += (1 - self.criterionSSIM(self.fake_Ts[i], self.real_T)) * np.power(sigma, iter_num - i - 1)
                self.loss_comp += self.criterionIdt((1 - self.rcmaps[i - 1]) * self.real_T + self.real_R, self.real_I) \
                                  * np.power(sigma, iter_num - i - 1)

                for j in range(self.fake_Ts[i].shape[0]):
                    real_I_r = torch.pow(self.real_I[j].unsqueeze(0), 2.2)
                    real_T_r = torch.pow(self.real_T[j].unsqueeze(0), 2.2)
                    T_r = torch.pow(self.fake_Ts[i][j].unsqueeze(0), 2.2)
                    R_r = torch.pow(self.fake_Rs[i][j].unsqueeze(0), 2.2)
                    if self.isNatural[j] == 0:
                        alpha = self.alpha[j]
                        self.loss_res += self.criterionIdt(real_I_r, (alpha * T_r + R_r)) \
                                         * np.power(sigma, iter_num - i)
                        self.loss_idt_R += self.criterionIdt(R_r + real_T_r * alpha, real_I_r) \
                                           * np.power(sigma, iter_num - i)
                        self.loss_comp += self.criterionIdt((1 - self.rcmaps[i - 1][j].unsqueeze(0))
                                                            * self.real_T[j].unsqueeze(0) + self.real_R[j].unsqueeze(0),
                                                            self.real_I[j].unsqueeze(0)) \
                                          * np.power(sigma, iter_num - i - 1)

        self.loss_mix = self.loss_idt_T * 0.16 + self.loss_ssim * 0.84
        self.loss_MP += self.criterionVgg(self.fake_T, self.real_T) + \
                        + 0.8 * self.criterionVgg(self.fake_T2, self.real_T2)\
                        + 0.6 * self.criterionVgg(self.fake_T4, self.real_T4)

        self.loss_CLIP_MSE += self.criterionCLIP_MSE(self.fake_Ts[3], self.real_T, self.res_model,
                                                     [1.0, 1.0, 1.0, 1.0, 0.5]) * 10
        prediction = torch.max(F.softmax(self.netLearn_Prompt(preprocess_feature(self.fake_Ts[3].detach(), self.clip_model), 0)), 1)[1]
        for i in range(self.fake_T.shape[0]):
            if prediction[i].item() == 2:
                self.early_stop_tags[i] == 0
            else:
                if self.early_stop_tags[i] == 1:
                    self.loss_CLIP += self.criterionCLIP(self.fake_Ts[3][i:i + 1], self.text_features, self.clip_model, prediction[i].item()) * 0.1
        # self.loss_CLIP_R += self.criterionCLIP_R(self.fake_Rs[3], self.text_features, self.clip_model) * 0.1
        # self.loss_CLIP_MSE_R += self.criterionCLIP_MSE(self.fake_Rs[3], self.real_R, self.res_model,
        #                                                [1.0, 1.0, 1.0, 1.0, 0.5])

        # self.loss_cont_B, self.loss_cont_R = self.calculate_MutualNCE_loss(self.fake_T, self.fake_R, self.real_T, self.real_R)

        self.loss_cont_B = self.loss_cont_B * 0.1
        self.loss_cont_R = self.loss_cont_R * 0.1

        # self.loss_G = self.criterionGAN(self.netD(self.fake_T), True) * 0.01  # L_adv: adversarial loss
        self.loss_T = self.loss_comp * 0.4 + self.loss_MP * 0.2 + self.loss_mix * 0.4 + self.loss_res * 0.4 + \
                      self.loss_CLIP_MSE + self.loss_CLIP
        self.loss_R = self.loss_idt_R * 0.4 + self.loss_CLIP_MSE_R
        self.loss = self.loss_T + self.loss_R

        self.loss.backward()

    # def backward_clip(self):
    #     self.loss_CLIP_model = 0.
    #     img_tensors = torch.cat([self.real_R, self.real_I, self.real_T], dim=0)
    #     labels = torch.cat([torch.zeros(self.real_R.shape[0]).long(), torch.ones(self.real_I.shape[0]).long(),
    #                         torch.ones(self.real_R.shape[0]).long() * 2], dim=0).to(img_tensors.device)
    #
    #     output = learn_prompt(img_tensors, 0)
    #     self.loss_CLIP_model += F.cross_entropy(output, label)
    #     self.loss_CLIP_model += self.criterionCLIPmodel(img_tensors, self.text_features, self.clip_model, labels)
    #     self.loss_CLIP_model.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # Pass invalid data
        if not self.trainFlag:
            self.trainFlag = True
            return

        embedding_prompt = self.netLearn_Prompt.module.embedding_prompt
        embedding_prompt.requires_grad = False
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in [" ".join(["X"] * 16)]])
        self.text_features = self.text_encoder(embedding_prompt, tokenized_prompts)

        self.set_requires_grad([self.netLearn_Prompt], False)
        self.set_requires_grad([self.netG_T], True)
        self.early_stop_tags = np.ones(self.opt.batch_size)
        for i in range(3):
            self.optimizer_G.zero_grad()
            self.forward()
            self.backward_G()
            self.optimizer_G.step()

        ############# update CLIP ############
        # self.set_requires_grad([self.clip_model], True)
        # self.set_requires_grad([self.netG_T], False)
        # # self.clip_model.train()
        # self.optimizer_clip.zero_grad()
        # self.backward_clip()
        # self.optimizer_clip.step()
        #
        # # self.clip_model.val()
        # self.set_requires_grad([self.clip_model], False)
        # embedding_prompt = self.netLearn_Prompt.module.embedding_prompt

        ############ update prompt ############
        embedding_prompt.requires_grad = True
        self.set_requires_grad([self.netG_T], False)
        self.optimizer_prompt.zero_grad()
        self.backward_prompt()
        self.optimizer_prompt.step()


        # self.set_requires_grad([self.netD], True)
        # self.optimizer_D.zero_grad()  # set D's gradients to zero
        # self.backward_D()  # calculate gradients for D
        # self.optimizer_D.step()

