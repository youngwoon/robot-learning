from main import run

import torch


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    run()
