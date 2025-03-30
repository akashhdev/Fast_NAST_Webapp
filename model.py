import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import random
import os

# Set reproducibility
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mean and std used during training
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

#######################################
# Model Definition and Helper Functions
#######################################

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, upsample=False, normalize=True, relu=True):
        super(ConvBlock, self).__init__()
        self.upsample = upsample
        # Wrap the convolution so that its parameters appear in a submodule "block"
        self.block = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        )
        # Normalization as a separate module ("norm")
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) if normalize else None
        self.relu = relu

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.block(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu:
            x = F.relu(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=True),
            ConvBlock(channels, channels, kernel_size=3, stride=1, normalize=True, relu=False)
        )

    def forward(self, x):
        return self.block(x) + x

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 32, kernel_size=9, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ConvBlock(128, 64, kernel_size=3, upsample=True),
            ConvBlock(64, 32, kernel_size=3, upsample=True),
            ConvBlock(32, 3, kernel_size=9, stride=1, normalize=False, relu=False)
        )

    def forward(self, x):
        return self.model(x)

#######################################
# Image Transformation Helpers
#######################################

def test_transform(image_size=256):
    """Transforms for test images"""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist())
    ])
    return transform

def denormalize(tensors):
    """Denormalizes image tensors using mean and std"""
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return tensors

def deprocess(image_tensor):
    """Converts image tensor to numpy array for display"""
    image_tensor = denormalize(image_tensor)[0]
    image_tensor *= 255
    image_np = torch.clamp(image_tensor, 0, 255).cpu().numpy().astype(np.uint8)
    image_np = image_np.transpose(1, 2, 0)
    return image_np

#######################################
# Test Image Function (as in Yash's repo)
#######################################

from torchvision.utils import save_image
import cv2
import matplotlib.pyplot as plt
from torch.autograd import Variable

def test_image(image_path, checkpoint_model, save_path):
    os.makedirs(os.path.join(save_path, "results"), exist_ok=True)
    transform_fn = test_transform()
    
    # Define model and load checkpoint
    transformer = TransformerNet().to(device)
    transformer.load_state_dict(torch.load(checkpoint_model, map_location=device))
    transformer.eval()
    
    # Prepare input
    image_tensor = Variable(transform_fn(Image.open(image_path))).to(device)
    image_tensor = image_tensor.unsqueeze(0)
    
    # Stylize image
    with torch.no_grad():
        stylized_image = denormalize(transformer(image_tensor)).cpu()
    
    # Save image
    fn = checkpoint_model.split('/')[-1].split('.')[0]
    output_filename = os.path.join(save_path, f"results/{fn}-output.jpg")
    save_image(stylized_image, output_filename)
    print("Image Saved!")
    