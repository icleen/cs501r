import os
import sys
import json
import numpy as np

import torch
from torchvision.utils import save_image

from generator import Generator
from descriminator import Descriminator


if __name__ == '__main__':
  config = sys.argv[1]
  with open(config, 'r') as f:
    config = json.load(f)

  train_info_path = config['model']['trainer_save_path']
  generator_path = config['model']['generator_save_path'].split('.pt')[0]
  descriminator_path = config['model']['descriminator_save_path'].split('.pt')[0]
  img_path = config['model']['image_save_path'].split('.png')[0]

  train_info = {}
  train_info = torch.load(train_info_path)
  itr = train_info['iter']
  dlosses = train_info['dlosses']
  glosses = train_info['glosses']
  g_optim = train_info['g_optimizer']
  d_optim = train_info['d_optimizer']


  z_size = config['train']['z_size']
  generator = Generator(z_size)
  descriminator = Descriminator()
  generator.load_state_dict(torch.load(
    str(generator_path + '_' + str(itr) + '.pt') ))
  descriminator.load_state_dict(torch.load(
    str(descriminator_path + '_' + str(itr) + '.pt') ))

  z1 = np.random.randn(z_size)
  z2 = np.random.randn(z_size)

  # interpolate
  steps = 8
  ad = (z2 - z1)/steps
  zs = [z1]
  for i in range(1,steps):
    zs.append(z1+(ad*i))
  zs.append(z2)
  

  #z = torch.zeros((batch_size, self.z_size), dtype=torch.float, device=self.device)
  #z.normal_()
  for i, z in enumerate(zs):
    z = torch.as_tensor(z, dtype=torch.float).unsqueeze(0)
    gen_img = generator(z)
    save_image(gen_img, 'generated_image_' + str(i) + '.png')
