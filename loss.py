import torch
import torch.nn as nn
from models.vgg19 import Vgg19
from torch.autograd import Variable
from torch.autograd import grad as torch_grad


def gradient_penalty(real, fake, D):
    eps = torch.rand(real.shape).cuda()
    x_std = torch.std(real, dim=[0, 1, 2, 3])

    fake = real + 0.5 * x_std * eps

    batch_size = real.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real)
    alpha = alpha.cuda()
    interpolated = alpha * real.data + (1 - alpha) * fake.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = D(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                            grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                            create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return 10 * ((gradients_norm - 1) ** 2).mean()


class AnimeLoss(object):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.vgg = Vgg19().cuda().eval()
        self.l1 = nn.L1Loss().cuda()
        self.huber = nn.SmoothL1Loss().cuda()

        self._rgb_to_yuv_kernel = torch.tensor([
            [0.299, -0.14714119, 0.61497538],
            [0.587, -0.28886916, -0.51496512],
            [0.114, 0.43601035, -0.10001026]
        ]).float().cuda()

    def rgb_to_yuv(self, image):
        image = (image + 1.0) / 2.0

        yuv_img = torch.tensordot(
            image,
            self._rgb_to_yuv_kernel,
            dims=([image.ndim - 3], [0]))

        return yuv_img

    def _vgg_loss(self, real, fake):
        x_vgg, y_vgg = self.vgg(real), self.vgg(fake)
        loss = self.l1(x_vgg, y_vgg)

        return loss

    def _color_loss(self, real, fake):
        real = self.rgb_to_yuv(real)
        fake = self.rgb_to_yuv(fake)

        # After convert to yuv, both images have channel last

        return (self.l1(real[:, :, :, 0], fake[:, :, :, 0]) +
                self.huber(real[:, :, :, 1], fake[:, :, :, 1]) +
                self.huber(real[:, :, :, 2], fake[:, :, :, 2]))

    def _style_loss(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        N, C, H, W = x_vgg.shape
        for n in range(N):
            phi_x = x_vgg[n]
            phi_y = y_vgg[n]
            phi_x = phi_x.reshape(C, H * W)
            phi_y = phi_y.reshape(C, H * W)
            G_x = torch.matmul(phi_x, phi_x.t()) / (C * H * W)
            G_y = torch.matmul(phi_y, phi_y.t()) / (C * H * W)
            loss += torch.sqrt(torch.mean((G_x - G_y) ** 2))

        return loss

    def _tv_loss(self, inputs):
        dh = inputs[:, :, :-1, :] - inputs[:, :, 1:, :]
        dw = inputs[:, :, :, :-1] - inputs[:, :, :, 1:]
        size_dh = dh.size().numel()
        size_dw = dw.size().numel()
        
        loss_h = torch.sum(dh ** 2) / size_dh * 2
        loss_w = torch.sum(dw ** 2) / size_dw * 2

        return  loss_h + loss_w

    def adv_loss_d_real(self, pred, adv_type='dragan'):
        if adv_type == 'lsgan':
            return torch.mean(torch.square(pred - 1.0))
        
        elif adv_type == 'dragan':
            return torch.mean(nn.BCEWithLogitsLoss()(torch.ones_like(pred), pred))

        raise ValueError(f'Do not support loss type {adv_type}')

    def adv_loss_d_fake(self, pred, adv_type='dragan'):
        if adv_type == 'lsgan':
            return torch.mean(torch.square(pred))

        elif adv_type == 'dragan':
            return torch.mean(nn.BCEWithLogitsLoss()(torch.zeros_like(pred), pred))

        raise ValueError(f'Do not support loss type {adv_type}')

    def adv_loss_g(self, pred, adv_type='dragan'):
        if adv_type == 'lsgan':
            return torch.mean(torch.square(pred - 1.0))

        elif adv_type == 'dragan':
            return torch.mean(nn.BCEWithLogitsLoss()(torch.ones_like(pred), pred))

        raise ValueError(f'Do not support loss type {adv_type}')

    def loss_init(self, real, generated):
        loss = self._vgg_loss(real, generated) * self.opt.con_weight
        return loss

    def loss_G(self, generated, real, generated_D, gray):
        loss_GAN = self.adv_loss_g(generated_D) * self.opt.g_adv_weight
        loss_VGG = self._vgg_loss(real, generated) * self.opt.con_weight
        loss_Color = self._color_loss(real, generated) * self.opt.color_weight
        loss_Style = self._style_loss(gray, generated) * self.opt.sty_weight
        loss_TV = self._tv_loss(generated) * self.opt.tv_weight

        loss = loss_GAN + loss_VGG + loss_Style + loss_TV + loss_Color

        print("G ", loss_GAN.item(), loss_VGG.item(), loss_Style.item(), loss_TV.item(), loss_Color.item())

        return loss
        
    def loss_D(self, generated_D, anime_D, anime_gray_D, anime_smooth_D):
        style_loss_D = self.adv_loss_d_real(anime_D)
        gray_loss_D = self.adv_loss_d_fake(anime_gray_D)
        fake_loss_D = self.adv_loss_d_fake(generated_D)
        blur_loss_D = self.adv_loss_d_fake(anime_smooth_D)

        loss = (style_loss_D * 1.2 + fake_loss_D * 1.2 + gray_loss_D * 1.5 + blur_loss_D * 0.8) * self.opt.d_adv_weight

        print("\nD ", style_loss_D.item(), gray_loss_D.item(), fake_loss_D.item(), blur_loss_D.item())

        return loss