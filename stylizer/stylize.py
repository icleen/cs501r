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


class StylizerLoss(nn.Module):
  def __init__(self, alpha, beta):
    super(StylizerLoss, self).__init__()
    self.alpha = alpha
    self.beta = beta

  def forward(self, content_loss, style_loss):
    return content_loss.mul(self.alpha) + style_loss.mul(self.beta)


class SquaredErrorLoss(nn.Module):
  def __init__(self, weight=None, size_average=None, ignore_index=-100,
               reduce=None, reduction='elementwise_mean'):
    self.__dict__.update(locals())
    super(SquaredErrorLoss, self).__init__()

  def forward(self, inp, targets):
    assert inp.size(0) == targets.size(0)
    return inp.sub(targets).pow(2).sum()/2


class GramLoss(nn.Module):
  def __init__(self, gramCalc):
    super(GramLoss, self).__init__()
    self.gramCalc = gramCalc

  def forward(self, inp, target_gram):
    nl = inp.size(0)*inp.size(1)
    ml = inp.size(2)*inp.size(3)
    in_gram = self.gramCalc(inp)
    assert in_gram.size() == target_gram.size()
    return in_gram.sub(target_gram).pow(2).sum().div(4*nl*nl*ml*ml)


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
    return gram


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


def main():

  load_and_normalize = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
  ])

  style_image = Image.open("style.png")
  style_image = load_and_normalize(np.array(style_image)).unsqueeze(0)

  content_image = Image.open("content.png")
  content_image = load_and_normalize(np.array(content_image)).unsqueeze(0)

  vgg_dict = {'conv4_1': 17, 'conv5_1': 24, 'conv2_2': 7, 'relu5_2': 27,
    'relu3_3': 15, 'relu3_1': 11, 'conv5_3': 28, 'conv3_2': 12, 'relu5_3': 29,
    'conv1_2': 2, 'relu3_2': 13, 'conv5_2': 26, 'conv4_3': 21, 'maxpool3': 16,
    'relu2_2': 8, 'relu4_1': 18, 'relu1_2': 3, 'conv3_1': 10, 'relu5_1': 25,
    'maxpool5': 30, 'conv2_1': 5, 'maxpool4': 23, 'conv1_1': 0, 'relu2_1': 6,
    'relu4_2': 20, 'relu4_3': 22, 'conv3_3': 14, 'maxpool1': 4, 'maxpool2': 9,
    'relu1_1': 1, 'conv4_2': 19}
  requested_vals = [vgg_dict['conv1_1'], vgg_dict['conv2_1'],
    vgg_dict['conv3_1'], vgg_dict['conv4_1'], vgg_dict['conv5_1']]
  # print(requested_vals)


  gramCalc = GramMatrix()

  vgg = VGGIntermediate(requested=requested_vals)
  # vgg.cuda()
  # layer = vgg(style_image.cuda())
  layer = vgg(style_image)
  style_activations = [layer[val] for val in requested_vals]
  style_grams = [gramCalc(act) for act in style_activations]
  # print('style_gram: {}'.format(style_grams[0].size()))
  w = 1.0/len(requested_vals)
  style_weights = [w for _ in requested_vals]
  w = None

  layer = vgg(content_image)
  content_activations = [layer[val] for val in requested_vals]

  # print( 'style_acts: {}, content_acts: {}'.format(len(style_activations), len(content_activations)) )
  # print('style:')
  # for act in style_activations:
  #   print('size: {}'.format(act.size()))
  # print('content:')
  # for act in content_activations:
  #   print('size: {}'.format(act.size()))

  input_img = content_image.clone()

  total_objective = StylizerLoss(1, 1e3)
  sel_objective = SquaredErrorLoss()
  gram_objective = GramLoss(gramCalc)
  optimizer = optim.Adam([input_img.requires_grad_()], lr=1e-4)

  iterations = 10
  for it in range(iterations):
    style_loss = torch.tensor(0.0)
    layer = vgg(input_img)
    input_activations = [layer[val] for val in requested_vals]

    for l, in_act in enumerate(input_activations):
      optimizer.zero_grad()

      style_loss.add_( gram_objective(in_act, style_grams[l]).mul(style_weights[l]) )
      # print( 'layer {} style loss: {}'.format(l+1, style_loss) )

      content_loss = sel_objective(in_act, content_activations[l])
      # print( 'layer {} content loss: {}'.format(l+1, content_loss) )

      tot_loss = total_objective(content_loss, style_loss)
      print( 'iter {} layer {} total loss: {}'.format(it+1, l+1, tot_loss) )

      tot_loss.backward(retain_graph=True)
      optimizer.step()

  # print('input_img.size(): {}'.format(input_img.size()))
  img = input_img.squeeze(0).detach().numpy()
  img = np.transpose(img, (1, 2, 0))
  img = img - np.min(img)
  img = img / np.max(img)
  # print(img.shape)
  out = Image.fromarray((img * 255).astype(np.uint8))
  out.save('output_image.png')


if __name__ == '__main__':
  main()
