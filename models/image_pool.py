import random

import torch


class ImagePool:
    """History buffer for generated images, matching the original CycleGAN logic."""

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.detach(), 0)
            if len(self.images) < self.pool_size:
                self.images.append(image)
                return_images.append(image)
            elif random.random() > 0.5:
                idx = random.randint(0, self.pool_size - 1)
                old = self.images[idx].clone()
                self.images[idx] = image
                return_images.append(old)
            else:
                return_images.append(image)
        return torch.cat(return_images, dim=0)

