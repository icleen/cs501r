import torch
import matplotlib.pyplot as plt
import sys

def main():
    if len(sys.argv) < 2:
        print( "Usage:", sys.argv[0], "log_file" )

    logf = sys.argv[1]
    info = {}
    info = torch.load(logf)

    print('valloss: {}'.format(info['valloss']))

    plt.plot(info['losses'])
    plt.savefig('loss_graph.png')


if __name__ == '__main__':
    main()
