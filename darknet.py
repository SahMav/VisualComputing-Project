from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from util import *
import cv2

#
# This code is written based on 'How to implement a YOLO v3 object detector from scratch in Pytorch' tutorial.
# By Ayoosha Kathuria
# https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
#


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


# makes blocks from the cfg
def parse_cfg(path):
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']           # no comments
    lines = [x.rstrip().lstrip() for x in lines]        # no whitespace

    list_of_blocks = []
    block = {}
    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                list_of_blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()        # block name
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()       # block content

    list_of_blocks.append(block)
    return list_of_blocks


# makes building blocks from list of blocks
def create_modules(blocks):
    network_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3       # RGB depth
    output_filters = []     # output filters of each block will be appended

    for index, block in enumerate(blocks[1:]):
        module = nn.Sequential()        # used when we have more than one module
        if block["type"] == "convolutional":
            activation = block["activation"]
            try:
                batch_normalize = int(block["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            stride = int(block["stride"])
            has_padding = int(block["pad"])

            if has_padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            if activation == "leaky":
                act = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), act)

        if block["type"] == "maxpool":
            kernel_size = int(block['size'])
            stride = int(block['stride'])
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            # for tiny Yolo implementation
            if kernel_size == 2 and stride == 1:
                module.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                module.add_module('MaxPool2d', maxpool)
            else:
                module = maxpool

        if block["type"] == "upsample":
            stride = int(block["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="bilinear")
            module.add_module("upsample_{}".format(index), upsample)

        if block["type"] == "route":
            block["layers"] = block["layers"].split(',')
            start = int(block["layers"][0])     # first element
            try:
                end = int(block["layers"][1])   # second element if it exists
            except:
                end = 0                         # second element if it doen't exist
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()                # empty layer because the computation is only a concat that happens in the forward pass
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end] # concat happens
            else:
                filters = output_filters[index + start]

        if block["type"] == "shortcut":
            shortcut = EmptyLayer()              # empty layer because the computation is only an addition
            module.add_module("shortcut_{}".format(index), shortcut)

        if block["type"] == "yolo":
            mask = block["mask"].split(',')
            mask = [int(m) for m in mask]
            anchors = block["anchors"].split(',  ')
            pairs = []
            for i in mask:
                pair = anchors[i].split(',')
                pair = [int(num) for num in pair]
                pairs.append(pair)
            anchors = pairs

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return network_info, module_list


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg("cfg/yolov3.cfg")
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}
        write = 0
        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)  # forward pass for convolutional and upsample layer

            if module_type == "route":      # forward pass for route layer
                layers = module["layers"]
                layers = [int(layer) for layer in layers]
                if len(layers) == 1:
                    if layers[0] > 0:
                        layers[0] -= i
                    x = outputs[layers[0] + i]
                else:
                    if layers[0] > 0:
                        layers[0] -= i
                    if layers[1] > 0:
                        layers[1] -= i
                    feature_map1 = outputs[layers[0] + i]
                    feature_map2 = outputs[layers[1] + i]
                    x = torch.cat((feature_map1, feature_map2), 1)

            if module_type == "shortcut":   # forward pass for shortcut layer
                f = int(module["from"])
                x = outputs[i - 1] + outputs[i + f]

            if module_type == "yolo":       # forward pass for yolo layer
                anchors = self.module_list[i][0].anchors
                input_dimension = int(self.net_info["height"])  # dimension from network information
                class_count = int(module["classes"])

                x = x.data
                x = predict_transform(x, input_dimension, anchors, class_count, CUDA)
                if not write:  # if no collector yet
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)  # if there is any detection initialized, we concat all
            outputs[i] = x

        return detections

    def load_weights(self, weights_file):
        f = open(weights_file, "rb")

        header = np.fromfile(f, dtype=np.int32, count=5)    # header info
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(f, dtype=np.float32)          # weights
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if batch_normalize:
                    bn = model[1]

                    # load weights
                    num_bn_biases = bn.bias.numel()
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # adjust to the models weights dimensions
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # add weights to the model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # load weights
                    num_biases = conv.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # adjust to the models weights dimensions
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # add weights to the model
                    conv.bias.data.copy_(conv_biases)

                # load convolutional layers weights
                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis,:,:,:]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = torch.from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_
#
# model = Darknet("cfg/yolov3.cfg")
# model.load_weights("yolov3.weights")
# inp = get_test_input()
# pred = model(inp, False)
# print (pred.size())





