import numpy as np
import tensorflow as tf
from PIL import Image

def load_image(path, shape=None):
    """Loads and preprocesses the image"""
    img = Image.open(path)
    if shape:
        img = img.resize((shape[1], shape[0]), Image.LANCZOS)
    return np.array(img).astype(np.float32)

def save_image(path, img):
    """Saves the output stylized image"""
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)

def get_img(image_path, img_size=None):
    """Loads image with optional resizing"""
    img = Image.open(image_path).convert('RGB')
    if img_size:
        img = img.resize((img_size, img_size), Image.LANCZOS)
    img = np.array(img).astype(np.float32)
    return img

def preprocess(image):
    """Prepares the image for the model"""
    return image - 255.0 / 2

def deprocess(image):
    """Converts model output to a displayable format"""
    return np.clip(image + 255.0 / 2, 0, 255)
