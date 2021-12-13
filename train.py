import os
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import AnimeDataset
from loss import AnimeLoss, gradient_penalty
from models.animeganv2.generator import Generator
from models.animeganv2.discriminator import Discriminator

import train_config as opt

opt.init_epoch = -1

model_G = Generator().cuda()
model_G.load_state_dict(torch.load('./checkpoint/model_7.pth')['G'])
model_D = Discriminator().cuda()

optimizer_G = optim.Adam(model_G.parameters(), lr=opt.g_lr, betas = (0.5, 0.999))
optimizer_D = optim.Adam(model_D.parameters(), lr=opt.d_lr, betas = (0.5, 0.999))

losses = AnimeLoss(opt)

train_dataset = AnimeDataset(root_path='./data', dataset='pixiv')
train_loader = DataLoader(
                    train_dataset, 
                    batch_size=opt.batch_size,
                    shuffle=True,
                )

for e in range(0, opt.epoch):
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, inputs in pbar:
        real = inputs['image'].cuda()
        style = inputs['style'].cuda()
        gray = inputs['gray'].cuda()
        smooth = inputs['smooth'].cuda()
        
        model_G.train()

        if e < opt.init_epoch:
            # init train generator
            optimizer_G.zero_grad()

            generated = model_G(real)
            loss_G = losses.loss_init(real, generated)
            loss_G.backward()
            optimizer_G.step()
            
            pbar.set_description(f"| G : {loss_G} |")

        else:
            # train discriminator
            optimizer_D.zero_grad()
            
            generated = model_G(real).detach()
            
            generated_D = model_D(generated)  
            anime_D = model_D(style)  
            anime_gray_D = model_D(gray)     
            anime_smooth_D = model_D(smooth)            

            GP = gradient_penalty(style, generated, D=model_D)

            loss_D = losses.loss_D(generated, real, generated_D, gray) + GP
            loss_D.backward()
            optimizer_D.step()

            # train generator
            optimizer_G.zero_grad()

            generated = model_G(real)
            generated_D = model_D(generated)

            loss_G = losses.loss_G(generated, real, generated_D, gray)
            loss_G.backward()
            optimizer_G.step()

            pbar.set_description(f"| D : {loss_D} | G : {loss_G} |")

        model_G.eval()
        with torch.no_grad():
            generated = model_G(real)
            torchvision.utils.save_image(torch.vstack([real, generated]), f'./{e}.png', normalize=True, nrow=6)

    if e < opt.init_epoch:
        torch.save(model_G.state_dict(), os.path.join(opt.checkpoint_path, f'init_model_{e}.pth'))
    else:
        torch.save({
            'D': model_D.state_dict(),
            'G': model_G.state_dict(),
        }, os.path.join(opt.checkpoint_path, f'model_{e}.pth'))

