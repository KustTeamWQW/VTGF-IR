import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from . import GANnet
from util import util


class IrenhanceModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):

        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_G', type=float, default=0.02, help='weight for loss_G_single')
            parser.add_argument('--lambda_Gg', type=float, default=0.8, help='weight for adversarial_loss')
            parser.add_argument('--lambda_Gp', type=float, default=0.1, help='weight for pixel_loss')
            parser.add_argument('--lambda_Gf', type=float, default=0.0, help='weight for feature_loss')
            parser.add_argument('--lambda_identity', type=float, default=0.68,
                                help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_rec_I', type=float, default=0.3, help='weight for loss_rec_I')
        parser.add_argument('--G_L', type=str, default='unet_trans_256', help='specify generator architecture')
        parser.add_argument('--G_J', type=str, default='resnet_9blocks', help='specify generator architecture')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_single', 'G', 'G_single', 'Adversarial', 'Pixel', 'Feature', 'rec_I', 'idt_J']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>

        if self.isTrain:
            self.visual_names = ['real_I', 'Lt', 'refine_J', 'rec_I',
                                 'rec_J', 'real_J', 'ref_real_J']
        else:
            self.visual_names = ['real_I', 'refine_J', 'rec_I', 'rec_J']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_L', 'G_J', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G_L', 'G_J']

        # define networks (both Generators and discriminators)
        self.G_L = networks.define_G(3, 3, opt.ngf, opt.G_L, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.G_J = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.G_J, opt.norm,
                                              not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators
            self.D = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert (opt.input_nc == opt.output_nc)
            self.fake_I_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_J_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionRec = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.feature_loss = GANnet.ContentLoss(
                    net_cfg_name="vgg19",
                    batch_norm=False,
                    num_classes=1000,
                    model_weights_path="",  # 空字符串表示使用预训练模型
                    feature_nodes=["features.35"],  # VGG19 的某个特征层
                    feature_normalize_mean=[0.485, 0.456, 0.406],
                    feature_normalize_std=[0.229, 0.224, 0.225]
                ).to(self.device)
            self.pixel_loss = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.G_L.parameters(), self.G_J.parameters()), lr=opt.lr,
                betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.real_I = input['infrared'].to(self.device)  # [-1, 1]
        self.image_paths = input['paths']
        if self.isTrain:
            self.real_J = input['visible'].to(self.device)  # [-1, 1]

    def forward(self):
        self.Lt = self.G_L(self.real_I)
        self.refine_J = self.G_J(self.real_I)
        self.rec_I = util.synthesize(self.refine_J, self.Lt)
        self.rec_J = util.reverse(self.real_I, self.Lt)

    def test(self):
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D(self):
        fake_J = self.fake_I_pool.query(self.refine_J)
        self.loss_D_single = self.backward_D_basic(self.D, self.real_J, fake_J)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_G = self.opt.lambda_G
        lambda_Gg = self.opt.lambda_Gg
        lambda_Gp = self.opt.lambda_Gp
        lambda_Gf = self.opt.lambda_Gf
        lambda_rec_I = self.opt.lambda_rec_I


        # Generator losses for rec_I and refine_J
        self.loss_Adversarial = self.criterionGAN(self.D(self.refine_J), True) * lambda_Gg
        self.loss_Pixel = self.pixel_loss(self.refine_J,self.real_J) * lambda_Gp
        self.loss_Feature = self.feature_loss(self.refine_J,self.real_J) * lambda_Gf
        self.loss_G_single = (self.loss_Adversarial + self.loss_Pixel + self.loss_Feature) * lambda_G

        # Reconstrcut loss
        self.loss_rec_I = self.criterionRec(self.real_I, self.rec_I) * lambda_rec_I

        # Identity loss
        self.ref_real_J = self.G_J(self.real_J)
        self.loss_idt_J = self.criterionIdt(self.ref_real_J, self.real_J) * lambda_idt

        # Total loss
        self.loss_G = self.loss_G_single + self.loss_rec_I + self.loss_idt_J

        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad(self.D, False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D
        self.set_requires_grad(self.D, True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D()  # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights
