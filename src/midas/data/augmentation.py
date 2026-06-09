import numpy as np
from PIL import Image
from torchvision import transforms
from skimage import color

class LABAugmentation:
    def __init__(self, t_range=(-0.3, 1.0), a_scale=10.0, b_scale=16.0, l_shift_range=(-8, 3), p=0.5):
        self.t_range = t_range
        self.a_scale = a_scale
        self.b_scale = b_scale
        self.l_shift_range = l_shift_range
        self.p = p

    def __call__(self, image: Image.Image):
        image_np = np.array(image).astype(np.float32) / 255.0
        lab = color.rgb2lab(image_np)

        t = np.random.uniform(*self.t_range)
        a_shift = t * self.a_scale
        b_shift = t * self.b_scale

        l_shift = np.random.uniform(*self.l_shift_range)

        lab[:, :, 0] = np.clip(lab[:, :, 0] + l_shift, 0, 100)
        lab[:, :, 1] = np.clip(lab[:, :, 1] + a_shift, -128, 127)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + b_shift, -128, 127)

        rgb = color.lab2rgb(lab)
        return Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))

augment_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomResizedCrop(224, scale=(0.65, 1.0)), 
                    transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(180),
                    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.4), 
                    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)), transforms.ColorJitter(brightness=0.2, contrast=0.2), 
                    LABAugmentation(t_range=(-0.15, 0.9), a_scale=10.0, b_scale=16.0, l_shift_range=(-3, 6), p=0.6), transforms.ToTensor(), 
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

standard_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])