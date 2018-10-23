import torch
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) < 2:
        print( "Usage:", sys.argv[0], "log_file" )

    logf = sys.argv[1]
    info = {}
    torch.load(logf, info)

    print(info.keys())


if __name__ == '__main__':
    main()
