import numpy as np
import torch
anchors = "10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326"
anchors = anchors.split(',  ')
pairs = []
for i in range(4):
    pair = anchors[i].split(',')
    pair = [int(num) for num in pair]
    pairs.append(pair)

anchors = torch.FloatTensor(pairs)
anchors = anchors.repeat(3 * 3, 1).unsqueeze(0)

grid = np.arange(13)
a,b = np.meshgrid(grid, grid)
x_offset = torch.FloatTensor(a).view(-1,1)
y_offset = torch.FloatTensor(b).view(-1,1)
x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, 3).view(-1, 2).unsqueeze(0)

print(anchors)
