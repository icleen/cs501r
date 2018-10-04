import sys
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
  logpath = sys.argv[1]

  with open(logpath, 'r') as f:
    lines = [line.strip().split(',') for line in f]

  trains = [line for line in lines if line[0] == 'train']
  vals = [line for line in lines if line[0] == 'valid']

  trainitrs = [int(t[1]) for t in trains]
  trainloss = [float(t[2]) for t in trains]

  valitrs = [int(t[1]) for t in vals]
  valloss = [float(t[2]) for t in vals]

  print('trainloss: {}, Valid IOU: {}'.format(np.min(trainloss), np.max(valloss)))

  plt.plot(trainitrs, trainloss, label='train loss per 20 iterations')
  plt.plot(valitrs, valloss, label='val IOU per 50 iterations')
  plt.scatter(trainitrs[np.argmin(trainloss)], np.min(trainloss), label='lowest loss')
  plt.scatter(valitrs[np.argmax(valloss)], np.max(valloss), label='highest IOU')
  plt.legend()
  plt.xlabel('iterations')
  plt.ylabel('loss')
  plt.savefig('loss_figure.png')
  # plt.show()


if __name__ == '__main__':
  main()
