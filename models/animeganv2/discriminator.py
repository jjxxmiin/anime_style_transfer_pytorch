import torch
import torch.nn as nn


class ConvNormLReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, bias=False, norm=True):
        
        pad_layer = {
            "zero":    nn.ZeroPad2d,
            "same":    nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError
        
        if norm:
            super(ConvNormLReLU, self).__init__(
                pad_layer[pad_mode](padding),
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias),
                nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            )   
        else:
            super(ConvNormLReLU, self).__init__(
                pad_layer[pad_mode](padding),
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias),
                nn.LeakyReLU(0.2, inplace=True)
            )


class Discriminator(nn.Module):
    def __init__(self, n_dis=3):
        super().__init__()
        in_ch = 32
        out_ch = 32

        self.layer = []

        self.layer.append(ConvNormLReLU(3, out_ch, norm=False))
        
        for _ in range(0, n_dis):
            self.layer.append(ConvNormLReLU(in_ch, out_ch * 2, stride=2, norm=False))
            self.layer.append(ConvNormLReLU(out_ch * 2, out_ch * 4))

            in_ch = out_ch * 4
            out_ch = out_ch * 2

        self.layer.append(ConvNormLReLU(in_ch, out_ch * 2))
        self.layer.append(nn.Conv2d(out_ch * 2, 1, kernel_size=3, stride=1, padding=1, bias=False))

        self.layer = nn.Sequential(*self.layer)

    def forward(self, input, gray=False):
        output = self.layer(input)

        return output


if __name__ == "__main__":
    model = Discriminator()

    input = torch.rand(1, 3, 256, 256)

    output = model(input)

    print(output.shape)