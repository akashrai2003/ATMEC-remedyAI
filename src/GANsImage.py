import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import os
import numpy as np
from torchvision.utils import save_image
import zipfile

class GeneratorChest(nn.Module):
    def __init__(self, opt):
        super(GeneratorChest, self).__init__()
        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

"""
Generate images from WGAN-GP for the chestXray Dataset
"""

class OptChest:
    n_epochs = 100
    batch_size = 128
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    n_cpu = 8
    latent_dim = 100
    img_size = 128
    channels = 1
    sample_interval = 600
    critic_iterations = 10 

optChest = OptChest()
generatorChest = GeneratorChest(optChest)
# Load the state dictionaries
generatorChest.load_state_dict(torch.load(r'C:\Users\akash\Desktop\Synthetic-Data-Generation-in-Medical-Applications\gans\wganc_generator.pth'))
# Set the models to evaluation mode
generatorChest.eval()
cuda = torch.cuda.is_available()
# If using CUDA
if cuda:
    generatorChest.cuda()

latent_dim = optChest.latent_dim
batch_size = optChest.batch_size
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
os.makedirs(r'A:\Downloads\Remedy.ai\generated_images\WGANGP_ch', exist_ok=True)
output_dir_chest = r'A:\Downloads\Remedy.ai\generated_images\WGANGP_ch'


def generate_images_gan_chest(
        num_images: int
):
    global batch_size, latent_dim
    batch_size = optChest.batch_size
    latent_dim = optChest.latent_dim
    # Generate images in batches
    for i in range(0, num_images, batch_size):
        batch_size = min(batch_size, num_images - i)
        z = Variable(Tensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        gen_imgs = generatorChest(z)

        # Save the generated images
        for j in range(gen_imgs.size(0)):
            save_image(gen_imgs[j], f'{output_dir_chest}/image_{i+j}.png', normalize=True)

    # Zip the generated images
    zip_filename = f'{output_dir_chest}\generated_images.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, _, files in os.walk(output_dir_chest):
            for file in files:
                zipf.write(os.path.join(root, file), file)

    # Clean up images after zipping
    for file in os.listdir(output_dir_chest):
        if file.endswith(".png"):
            os.remove(os.path.join(output_dir_chest, file))

    return zip_filename





class GeneratorKnee(nn.Module):
    def __init__(self, opt):
        super(GeneratorKnee, self).__init__()
        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

"""
Generate images from WGAN-GP for the chestXray Dataset
"""

class OptKnee:
    n_epochs = 200
    batch_size = 64
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    n_cpu = 8
    latent_dim = 100
    img_size = 128
    channels = 1
    sample_interval = 400

optKnee = OptKnee()
generatorKnee = GeneratorKnee(optKnee)
# Load the state dictionaries
generatorKnee.load_state_dict(torch.load(r'A:\Downloads\Remedy.ai\models\wgangp_generator.pth'))
# Set the models to evaluation mode
generatorKnee.eval()

cuda = torch.cuda.is_available()
# If using CUDA
if cuda:
    generatorKnee.cuda()

latent_dim = optKnee.latent_dim
batch_size = optKnee.batch_size
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Create a directory to save the generated images
os.makedirs(r'A:\Downloads\Remedy.ai\generated_images\WGANGP_knee', exist_ok=True)
output_dir_knee = r'A:\Downloads\Remedy.ai\generated_images\WGANGP_knee'


def generate_images_gan_knee(
        num_images: int,
):
    global batch_size, latent_dim
    batch_size = optChest.batch_size
    latent_dim = optChest.latent_dim
    # Generate images in batches
    for i in range(0, num_images, batch_size):
        batch_size = min(batch_size, num_images - i)
        z = Variable(Tensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        gen_imgs = generatorKnee(z)

        # Save the generated images
        for j in range(gen_imgs.size(0)):
            save_image(gen_imgs[j], f'{output_dir_knee}/image_{i+j}.png', normalize=True)

    # Zip the generated images
    zip_filename = f'{output_dir_knee}\generated_images.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, _, files in os.walk(output_dir_knee):
            for file in files:
                zipf.write(os.path.join(root, file), file)

    # Clean up images after zipping
    for file in os.listdir(output_dir_knee):
        if file.endswith(".png"):
            os.remove(os.path.join(output_dir_knee, file))

    return zip_filename
