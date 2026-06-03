from torchvision import transforms


def get_transform(load_size=286, image_size=256, phase="train"):
    """Build image transforms for CycleGAN train/test phases."""
    if phase == "train":
        ops = [
            transforms.Resize((load_size, load_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    else:
        ops = [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    return transforms.Compose(ops)

