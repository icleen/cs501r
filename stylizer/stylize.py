import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from torch.nn.parameter import Parameter
import torchvision.models as models

import json
import gc
import sys
import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# import pdb
# from tqdm import tqdm


class GramLoss(nn.Module):
  def __init__(self, gramCalc):
    super(GramLoss, self).__init__()
    self.gramCalc = gramCalc

  def forward(self, inp, target_gram):
    in_gram = self.gramCalc(inp)
    return F.mse_loss(in_gram, target_gram)


class GramMatrix(nn.Module):
  def __init__(self):
    super(GramMatrix, self).__init__()

  def forward(self, mat):
    """batch_size, N_filters, height, width"""
    bl, nl, hl, wl = mat.size()
    """
    You collapse it so that you can just to a basic matrix multiplication
    and it's easy
    """
    collapsed = mat.view(bl * nl, hl * wl)
    gram = torch.mm(collapsed, collapsed.t())
    return gram.div(bl * nl * hl * wl)


class VGGIntermediate(nn.Module):
  def __init__(self, requested=[]):
    super(VGGIntermediate, self).__init__()

    self.intermediates = {}
    self.vgg = models.vgg16(pretrained=True).features.eval()
    for i, m in enumerate(self.vgg.children()):
        if isinstance(m, nn.ReLU):   # we want to replace the relu functions with in place functions.
          m.inplace = False          # the model has a hard time going backwards on the in place functions.

        if i in requested:
          def curry(i):
            def hook(module, input, output):
              self.intermediates[i] = output
            return hook
          m.register_forward_hook(curry(i))

  def forward(self, x):
    self.vgg(x)
    return self.intermediates


def export_img(img_tensor, img_path):
  if torch.cuda.is_available():
    img = img_tensor.squeeze(0).detach().cpu().numpy()
  else:
    img = img_tensor.squeeze(0).detach().numpy()
  img = np.transpose(img, (1, 2, 0))
  img = img - np.min(img)
  img = img / np.max(img)
  out = Image.fromarray((img * 255).astype(np.uint8))
  out.save(img_path)


def main():

  out_img_file = 'output_image'
  content_path = "content.png"
  style_path = "style.png"
  if len(sys.argv) > 1:
    out_img_file = sys.argv[1].split('.png')[0]
  if len(sys.argv) > 2:
    content_path = sys.argv[2]
  if len(sys.argv) > 3:
    style_path = sys.argv[3]

  load_and_normalize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])

  style_image = Image.open(style_path)
  style_image = load_and_normalize(np.array(style_image)).unsqueeze(0)

  content_image = Image.open(content_path)
  content_image = load_and_normalize(np.array(content_image)).unsqueeze(0)

  vgg_dict = {'conv4_1': 17, 'conv5_1': 24, 'conv2_2': 7, 'relu5_2': 27,
    'relu3_3': 15, 'relu3_1': 11, 'conv5_3': 28, 'conv3_2': 12, 'relu5_3': 29,
    'conv1_2': 2, 'relu3_2': 13, 'conv5_2': 26, 'conv4_3': 21, 'maxpool3': 16,
    'relu2_2': 8, 'relu4_1': 18, 'relu1_2': 3, 'conv3_1': 10, 'relu5_1': 25,
    'maxpool5': 30, 'conv2_1': 5, 'maxpool4': 23, 'conv1_1': 0, 'relu2_1': 6,
    'relu4_2': 20, 'relu4_3': 22, 'conv3_3': 14, 'maxpool1': 4, 'maxpool2': 9,
    'relu1_1': 1, 'conv4_2': 19}
  style_vals = [vgg_dict['conv1_1'], vgg_dict['conv2_1'],
    vgg_dict['conv3_1'], vgg_dict['conv4_1'], vgg_dict['conv5_1']]
  content_vals = [vgg_dict['conv4_2']]

  gramCalc = GramMatrix()

  vgg = VGGIntermediate(requested=(style_vals+content_vals))
  if torch.cuda.is_available():
    vgg.cuda()
    layer = vgg(style_image.cuda())
  else:
    layer = vgg(style_image)
  style_activations = [layer[val].clone() for val in style_vals]
  style_grams = [gramCalc(act) for act in style_activations]
  w = 1.0/len(style_vals)
  style_weights = [w for _ in style_vals]
  w = None

  if torch.cuda.is_available():
    layer = vgg(content_image.cuda())
  else:
    layer = vgg(content_image)
  content_activations = [layer[val].clone() for val in content_vals]

  input_img = content_image.clone()
  if torch.cuda.is_available():
    input_img = input_img.cuda()

  alpha, beta = 1e-1, 1e5
  tot_loss = 0.0
  content_objective = F.mse_loss
  style_objective = GramLoss(gramCalc)
  optimizer = optim.Adam([input_img.requires_grad_()], lr=0.1)

  iterations = 10000
  print_interval = iterations / 100
  img_interval = iterations / 10
  for it in range(iterations):
    optimizer.zero_grad()

    style_loss = torch.tensor(0.0)
    content_loss = torch.tensor(0.0)
    if torch.cuda.is_available():
      style_loss = style_loss.cuda()
      content_loss = content_loss.cuda()

    layer = vgg(input_img)
    input_acts = [layer[val] for val in style_vals]
    for l, in_act in enumerate(input_acts):
      style_loss += style_objective(in_act, style_grams[l]).mul(style_weights[l])

    input_acts = [layer[val] for val in content_vals]
    for l, in_act in enumerate(input_acts):
      content_loss += content_objective(in_act, content_activations[l])

    # print( 'iter {} layer {} style loss: {}'.format(it+1, l+1, style_loss) )
    # print( 'iter {} layer {} content loss: {}'.format(it+1, l+1, content_loss) )
    tot_loss = (alpha * content_loss) + (beta * style_loss)

    if it % print_interval == 0 and (l == 0 or l == 4):
      print( 'iter {} layer {} total loss: {}'.format(it, l+1, tot_loss) )

    if it % img_interval == 0 and (l == 0 or l == 4):
      export_img(input_img, out_img_file+'_'+str(it)+'_temp.png')

    tot_loss.backward(retain_graph=True)
    optimizer.step()

  print( 'iter {} layer {} total loss: {}'.format(it+1, l+1, tot_loss) )
  export_img(input_img, out_img_file+'.png')


if __name__ == '__main__':
  main()
