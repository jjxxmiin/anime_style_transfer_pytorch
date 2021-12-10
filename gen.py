import torch
from torchvision import utils
from models.stylegan import Generator
from tqdm import tqdm


def main():
    g_ema = Generator(
        1024, 512, 8, channel_multiplier=2
    ).to("cuda")

    g_ema.load_state_dict(torch.load('./checkpoint/stylegan2-ffhq-config-f.pt')["g_ema"], strict=False)
    g_ema.eval()

    with torch.no_grad():
        for i in tqdm(range(0, 10000)):
            sample_z = torch.randn(1, 512, device="cuda")

            sample, _ = g_ema(
                [sample_z], 
                truncation=1, 
                truncation_latent=None
            )

            utils.save_image(
                sample,
                f"./data/real/{i}.png",
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


if __name__ == "__main__":
    main()