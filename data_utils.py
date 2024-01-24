import medmnist
from torchvision import transforms

def get_medmnist_info(config):
    info = medmnist.INFO[config["data_flag"]]
    DataClass = getattr(medmnist, info["python_class"])
    return info, DataClass

def get_transforms():
    return transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
