import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.autograd import Variable
from skimage.color import rgb2yuv


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, layids=None):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()

        self.criterion = nn.L1Loss().cuda()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * \
                self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            N, C, H, W = x_vgg[i].shape
            for n in range(N):
                phi_x = x_vgg[i][n]
                phi_y = y_vgg[i][n]
                phi_x = phi_x.reshape(C, H * W)
                phi_y = phi_y.reshape(C, H * W)
                G_x = torch.matmul(phi_x, phi_x.t()) / (C * H * W)
                G_y = torch.matmul(phi_y, phi_y.t()) / (C * H * W)
                loss += torch.sqrt(torch.mean((G_x - G_y) ** 2)
                                   ) * self.weights[i]
        return loss


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(
                    real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(
                    fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class Color_Loss(nn.Module):
    def __init__(self):
        super(Color_Loss, self).__init__()

        self.l1_loss = nn.L1Loss()
        self.huber_loss = nn.SmoothL1Loss()

    def _convert(self, input_, type_):
        return {
            'float': input_.float(),
            'double': input_.double(),
        }.get(type_, input_)

    def rgb_to_yuv(self, input_):
        to_squeeze = (input_.dim() == 3)
        device = input_.device
        input_ = input_.cpu()
        input_ = self._convert(input_, type_='')

        if to_squeeze:
            input_ = input_.unsqueeze(0)

        input_ = input_.permute(0, 2, 3, 1).detach().numpy()
        transformed = rgb2yuv(input_)
        output = torch.from_numpy(transformed).float().permute(0, 3, 1, 2)
        
        if to_squeeze:
            output = output.squeeze(0)
        
        output = self._convert(output, type_='')

        return output.to(device)

    def __call__(self, real, fake):
        real = self.rgb_to_yuv(real)
        fake = self.rgb_to_yuv(fake)

        y = self.l1_loss(real[:, 0, :, :], fake[:, 0, :, :])
        u = self.huber_loss(real[:, 1, :, :], fake[:, 1, :, :])
        v = self.huber_loss(real[:, 2, :, :], fake[:, 2, :, :])

        return y + u + v


class Variation_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, inputs):
        dh = inputs[:, :, :-1, :] - inputs[:, :, 1:, :]
        dw = inputs[:, :, :, :-1] - inputs[:, :, :, 1:]
        size_dh = dh.size().numel()
        size_dw = dw.size().numel()
        
        loss_h = torch.sum(dh ** 2) / size_dh * 2
        loss_w = torch.sum(dw ** 2) / size_dw * 2

        return  loss_h + loss_w


if __name__ == "__main__":
    a = torch.rand(1, 3, 256, 256)
    b = torch.rand(1, 3, 256, 256)
    
    print(Color_Loss()(a, b))
    print(Variation_Loss()(a))

