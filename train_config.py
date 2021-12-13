
epoch = 101
init_epoch = 10
batch_size = 6
save_freq = 1

init_lr = 2e-4
g_lr = 2e-5
d_lr = 4e-5
ld = 10.0

g_adv_weight = 300.0
d_adv_weight = 300.0

con_weight = 1.5
color_weight = 10.0 
sty_weight = 2.5
tv_weight = 1.

training_rate = 1
gan_tyle = 'lsgan'

img_size = [256, 256]
img_ch = 3

n_dis = 3

checkpoint_path = './checkpoint'