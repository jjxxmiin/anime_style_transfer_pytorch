import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm
from dataset import AnimeDataset
from loss import VGGPerceptualLoss, GANLoss, StyleLoss, Color_Loss, Variation_Loss
from models.animeganv2.generator import Generator
from models.animeganv2.discriminator import Discriminator

import train_config as opt

model_G = Generator().cuda()
model_D = Discriminator().cuda()

optimizer_init = optim.Adam(model_G.parameters(), lr=opt.init_lr, betas = (0.5, 0.999))
optimizer_G = optim.Adam(model_G.parameters(), lr=opt.g_lr, betas = (0.5, 0.999))
optimizer_D = optim.Adam(model_D.parameters(), lr=opt.d_lr, betas = (0.5, 0.999))

criterionStyle = StyleLoss().cuda()
criterionVGG = VGGPerceptualLoss().cuda()
criterionL1 = nn.L1Loss().cuda()
criterionADV = GANLoss(target_real_label=0.9, target_fake_label=0.1).cuda()
criterionColor = Color_Loss().cuda()
criterionTV = Variation_Loss().cuda()

train_dataset = AnimeDataset(root_path='./data')
train_loader = DataLoader(
                    train_dataset, 
                    batch_size=opt.batch_size
                )

for e in range(0, opt.epoch):
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, inputs in pbar:
        real = inputs['image'].cuda()
        style = inputs['style'].cuda()
        gray = inputs['gray'].cuda()
        smooth = inputs['smooth'].cuda()
        
        if e < opt.init_epoch:
            # init train generator
            optimizer_init.zero_grad()
            generated = model_G(real)
            loss_G = criterionVGG(real, generated) * opt.con_weight
            loss_G.backward()
            optimizer_init.step()
            
            pbar.set_description(f"| G : {loss_G} |")

        else:
            # train discriminator
            optimizer_D.zero_grad()
            
            generated = model_G(real)

            style_D = model_D(style)  
            gray_D = model_D(gray)
            generated_D = model_D(generated.detach())       
            smooth_D = model_D(smooth)            

            style_loss_D = criterionADV(style_D, target_is_real=True)
            gray_loss_D = criterionADV(gray_D, target_is_real=False)
            fake_loss_D = criterionADV(generated_D, target_is_real=False)
            blur_loss_D = criterionADV(smooth_D, target_is_real=False)

            loss_D = 1.7 * style_loss_D + 1.7 * fake_loss_D + 1.7 * gray_loss_D + 1.0 * blur_loss_D
            loss_D *= opt.d_adv_weight
            loss_D.backward()
            optimizer_D.step()

            # train generator
            optimizer_G.zero_grad()
            fake_D = model_D(generated)
            loss_GAN = criterionADV(fake_D, target_is_real=True) * opt.g_adv_weight
            loss_VGG = criterionVGG(real, generated) * opt.con_weight
            loss_Style = criterionStyle(gray, generated) * opt.sty_weight
            loss_TV = criterionTV(generated) * opt.tv_weight
            loss_Color = criterionColor(real, generated) * opt.color_weight
            
            loss_G = loss_GAN + loss_VGG + loss_Style + loss_TV + loss_Color
            loss_G.backward()
            optimizer_G.step()

            pbar.set_description(f"| D : {loss_D} | G : {loss_G} |")

        torchvision.utils.save_image(generated, './test.png', normalize=True)

    if e < opt.init_epoch:
        torch.save(model_G.state_dict(), os.path.join(opt.checkpoint_path, f'init_model_{e}.pth'))
    else:
        torch.save({
            'D': model_D.state_dict(),
            'G': model_G.state_dict(),
        }, os.path.join(opt.checkpoint_path, f'model_{e}.pth'))

    print(loss_D, loss_G)

