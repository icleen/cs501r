import os
import json


if __name__ == '__main__':
  config = sys.argv[1]
  with open(config, 'r') as f:
    config = json.load(f)

  train_info_path = config['model']['trainer_save_path']
  generator_path = config['model']['generator_save_path'].split('.pt')[0]
  descriminator_path = config['model']['descriminator_save_path'].split('.pt')[0]
  img_path = config['model']['image_save_path'].split('.png')[0]

  train_info = {}
  train_info = torch.load(self.train_info_path)
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

  z1 = np.random.randn((1, z_size))
  z2 = np.random.randn((1, z_size))

  # interpolate
  interps = np.interp(z1, z1[:2], z2[:2])
  print(interps)

  gen_img = generator(z1)
  save_image(gen_img, 'generated_image1.png'))
  gen_img = generator(z2)
  save_image(gen_img, 'generated_image2.png'))
