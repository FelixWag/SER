from torchvision import transforms

def transform():
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])