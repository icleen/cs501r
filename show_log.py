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

  # plt.plot(range(len(trains)), trains[:][2], label='train loss per 20 iterations')
  plt.plot(trainitrs, trainloss, label='train loss per 20 iterations')
  # plt.plot(vals[:][1], vals[:][2], label='validation IOU per 50 iterations')
  plt.plot(valitrs, valloss, label='val IOU per 50 iterations')
  plt.legend()
  plt.xlabel('iterations')
  plt.ylabel('loss')
  plt.savefig('loss_figure.png')
  plt.show()


if __name__ == '__main__':
  main()
