import torch
import sys

sd = torch.load(sys.argv[1], 'cpu')
print(f"Best loss :{sd['best_loss']:.4f}@{sd['epoch']}")
