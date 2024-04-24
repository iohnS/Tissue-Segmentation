import torch
import random as rnd
from monai.metrics import MeanIoU, DiceMetric
from monai.losses import DiceLoss

pairs = []
for i in range(10):
    tensor1 = torch.empty(1, 1, 256, 256)
    tensor2 = torch.empty(1, 1, 256, 256)
    for i in range(256):
        for j in range(256):
            if rnd.random() < 0.5:
                tensor1[0, 0, i, j] = 0
            else:
                tensor1[0, 0, i, j] = 1
                
            if rnd.random() < 0.5:
                tensor2[0, 0, i, j] = 0
            else:
                tensor2[0, 0, i, j] = 1
    pairs.append((tensor1, tensor2))

zeros = torch.zeros(1, 1, 256, 256)
ones = torch.ones(1, 1, 256, 256)

miou = MeanIoU(ignore_empty=False)
dm = DiceMetric(ignore_empty=False)
dl = DiceLoss()

for pair in pairs:
    print(miou(y_pred=pair[0], y=pair[1]))
    print(dm(y_pred=pair[0], y=pair[1]))
    print(dl(pair[0], pair[1]))
    print("-" * 10)