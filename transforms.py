import torchvision.transforms as transforms


IMAGE_SIZE=(112, 184) # H x W

train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=1, hue=0.1),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

test_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])