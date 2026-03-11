import torch
import lpips
from kornia import augmentation
from kornia.constants import SamplePadding

class SDLoss(torch.nn.Module):
    def __init__(self, device, use_aug=True):
        super().__init__()
        self.device = device
        self.use_aug = use_aug
        self.lpips_fn = lpips.LPIPS(net='vgg').to(device)
        if self.use_aug:
            self.aug = augmentation.AugmentationSequential(
                augmentation.RandomHorizontalFlip(p=0.5),
                augmentation.RandomAffine(degrees=5, translate=None, p=0.5, padding_mode=SamplePadding.REFLECTION),
                augmentation.RandomAffine(degrees=0, translate=(0.01, 0.01), p=0.5, padding_mode=SamplePadding.REFLECTION),
                random_apply=2,
                data_keys=["input", "input"]
            ).to(device)

    def forward(self, gen_img, gen_img_ema):
        if self.use_aug:
            gen_img, gen_img_ema = self.aug(gen_img, gen_img_ema)
        loss = self.lpips_fn(gen_img * 2 - 1, gen_img_ema * 2 - 1).mean()
        return loss