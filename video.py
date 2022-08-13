from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
import argparse
import os
import os.path as osp
from darknet import Darknet
import pickle as pkl
import pandas as pd
import random
import math

#
# This code is written based on 'How to implement a YOLO v3 object detector from scratch in Pytorch' tutorial.
# By Ayoosha Kathuria
# https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
#

def arg_parse():

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')
    parser.add_argument("--bs", dest="bs", help="Batch size", default=1)
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.6)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help=
    "Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help=
    "weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument("--video", dest="videofile", help="Video file to     run detection on", default="video.avi",
                        type=str)

    return parser.parse_args()


args = arg_parse()
batch_size = int(args.bs)
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80
classes = load_classes("coco.names")

# Set up the neural network
print("Loading network.....")
model = Darknet(args.cfgfile)
model.load_weights(args.weightsfile)
print("Network successfully loaded")

model.net_info["height"] = args.reso
inp_dim = int(model.net_info["height"])
assert inp_dim % 32 == 0
assert inp_dim > 32


# Set the model in evaluation mode
model.eval()

# the list of calibration points
mouse_points = []

# Set 7 points for calibration, first 4 on the four corners of the perspective area and the next 3 for showing distance limits
def set_calibration_points(event, x, y, s, p):
    global mouse_points
    if event == cv2.EVENT_LBUTTONDOWN:
        new_point = (x, y)
        if len(mouse_points) < 4:
            cv2.circle(image, new_point, 3, (0, 255, 255), 2)
        if 4 <= len(mouse_points) < 7:
            cv2.circle(image, new_point, 3, (0, 0, 255), 2)
        if 'mouse_points' not in globals():
            mouse_points = []
        mouse_points.append(new_point)

# change perspective of the initial frame
def change_perspective(frame, mouse_points):
    corner_points = np.array(mouse_points[0:4])
    scale_points = np.array(mouse_points[4:7])
    h, w, c = frame.shape

    points = []
    for i in range(w):
        for j in range(h):
            points.append([i, j])
    # h = 700
    # w = 400

    corners_transformed = [[0, h], [w, h], [w, 0], [0, 0]]
    perspective_transform = cv2.getPerspectiveTransform(np.float32(corner_points), np.float32(corners_transformed))

    # scales_transformed = np.matmul(perspective_transform, scale_points)
    transformed = cv2.perspectiveTransform(np.float32([points]), perspective_transform)[0]
    mask = np.zeros((1000, 2200, 3), np.uint8)
    mask[:, :] = [255, 255, 255]
    # print(transformed)
    for i in range(len(transformed)):
        col = int(i / h)
        row = i % h
        mask[int(transformed[i][1]), 1250 + int(transformed[i][0])] = frame[row, col]
    resized = cv2.resize(mask, (500, 1000))
    cv2.imwrite('transformed.jpg', resized)


# for each frame apply the transform and check violations
def create_birds_eye_view(frame, mouse_points, output, bird_view_video):
    corner_points = np.array(mouse_points[0:4])
    scale_points = np.array(mouse_points[4:7])

    h, w, c = frame.shape

    h = 600
    w = 400
    # showing new corners of the rectangle for the top-down view
    corners_transformed = [[0, h], [w, h], [w, 0], [0, 0]]
    perspective_transform = cv2.getPerspectiveTransform(np.float32(corner_points), np.float32(corners_transformed))

    # scales_transformed = np.matmul(perspective_transform, scale_points)
    scales_transformed = cv2.perspectiveTransform(np.float32([scale_points]), perspective_transform)[0]
    first_point = scales_transformed[0]
    vertical_point = scales_transformed[2]
    horizontal_point = scales_transformed[1]

    vertical_limit = np.sqrt(np.dot((vertical_point - first_point), (vertical_point - first_point)))
    horizontal_limit = np.sqrt(np.dot((horizontal_point - first_point), (horizontal_point - first_point)))

    # applying transforms to bounding boxes in the frame
    real_locations = []
    transformed_locations = []
    first_corners = []
    second_corners = []
    for tensor in output:
        c1 = tuple(tensor[1:3].int())
        c2 = tuple(tensor[3:5].int())
        center_x = (c1[0] + c2[0]) / 2
        center_y = (c1[1] + c2[1]) / 2

        if tensor[-1] == 0:
            x = tensor[3].int()
            y = tensor[4].int()
            point = np.array([[x, y]])
            transformed_point = cv2.perspectiveTransform(np.float32([point]), perspective_transform)[0]
            transformed_locations.append(transformed_point[0])
            real_locations.append((center_x, center_y))
            first_corners.append(c1)
            second_corners.append(c2)

    # detect violations
    person_flag = ['safe' for _ in range(len(transformed_locations))]
    distance_mat = np.zeros((len(transformed_locations), len(transformed_locations)))
    for i in range(len(transformed_locations)):
        for j in range(i):
            point_1 = transformed_locations[i]
            point_2 = transformed_locations[j]
            horizontal_distance_vec = point_2[0] - point_1[0]
            vertical_distance_vec = point_2[1] - point_1[1]
            horizontal_distance = np.sqrt(np.dot(horizontal_distance_vec, horizontal_distance_vec))
            vertical_distance = np.sqrt(np.dot(vertical_distance_vec, vertical_distance_vec))
            # distance_mat[i][j] = distance
            if horizontal_distance < horizontal_limit and vertical_distance < vertical_limit:
                person_flag[i] = 'risk'
                person_flag[j] = 'risk'

    # visualize violations
    for i in range(len(person_flag)):
        if person_flag[i] == 'safe':
            cv2.rectangle(frame, first_corners[i], second_corners[i], (20, 255, 0), 2)
        # cv2.circle(frame, real_locations[i], 5, (20, 255, 0), 2)
        if person_flag[i] == 'risk':
            cv2.rectangle(frame, first_corners[i], second_corners[i], (0, 0, 255), 2)
            # cv2.circle(frame, real_locations[i], 5, (0, 0, 255), 2)

    transformed_frame = create_bird_view_image(transformed_locations, 600, 400, person_flag)
    bird_view_video.write(transformed_frame)
    return person_flag

# for each frame create a new frame showing people in the top-down view
def create_bird_view_image(transformed_locations, height, width, person_flag):
    # print((transformed_locations))
    frame = np.zeros((height, width, 3), np.uint8)
    frame[:, :] = [255, 255, 255]

    for j in range(len(transformed_locations)):
        location = transformed_locations[j]
        if person_flag[j] == 'safe':
            cv2.circle(frame, (int(location[0]), int(location[1])), 5, (0, 255, 0), 2)
        if person_flag[j] == 'risk':
            cv2.circle(frame, (int(location[0]), int(location[1])), 5, (0, 0, 255), 2)

    # cv2.imshow('bird view', frame)
    return frame

# compute the two-by-two distance in 2D Euclidean and visualize violations(only for reference)
def compute_euclidean_distance(output, frame):
    for i in range(len(output)):
        tensor = output[i]
        if tensor[-1] == 0:
            first_corner_x = int(tensor[1])
            fist_corner_y = int(tensor[2])
            second_corner_x = int(tensor[3])
            second_corner_y = int(tensor[4])
            center_x = int((first_corner_x + second_corner_x) / 2)
            center_y = int((fist_corner_y + second_corner_y) / 2)
            cv2.circle(frame, (center_x, center_y), 4, (255, 255, 255), 2)

            for j in range(i):
                other_tensor = output[j]
                if other_tensor[-1] == 0:
                    first_corner_x_other = int(other_tensor[1])
                    fist_corner_y_other = int(other_tensor[2])
                    second_corner_x_other = int(other_tensor[3])
                    second_corner_y_other = int(other_tensor[4])
                    center_x_other = int((first_corner_x_other + second_corner_x_other) / 2)
                    center_y_other = int((fist_corner_y_other + second_corner_y_other) / 2)
                    distance = math.sqrt((center_x - center_x_other) ** 2 + (center_y - center_y_other) ** 2)
                    if distance < 100:
                        cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), 2)
                        cv2.circle(frame, (center_x_other, center_y_other), 4, (0, 0, 255), 2)
                        cv2.line(frame, (center_x, center_y), (center_x_other, center_y_other), (0, 0, 255), 2)

    return frame


cv2.namedWindow('first frame')
cv2.setMouseCallback('first frame', set_calibration_points)
np.random.seed(42)


def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    center_x = (c1[0] + c2[0]) / 2
    center_y = (c1[1] + c2[1]) / 2
    img = results
    cls = int(x[-1])
    # if cls == 0:

    return img


# Detection phase

videofile = 'pedestrian.mp4'  # or path to the video file.

cap = cv2.VideoCapture(videofile)

# cap = cv2.VideoCapture(0)  for webcam

assert cap.isOpened(), 'Cannot capture source'

# bird_view_video = None

frames = -1
start = time.time()
skip = 1
global image
ret, frame = cap.read()

h, w, c = frame.shape
fps = 5
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
bird_view_video = cv2.VideoWriter("bird_view_test.mp4", fourcc, fps, (400, 600))
# output_video = cv2.VideoWriter("output_view_test.mp4", fourcc, fps, (h, w))

while cap.isOpened():
    # calibration happens
    if frames == -1:
        image = frame
        cv2.imshow('first frame', frame)
        # output_video.write(frame)
        cv2.waitKey(1)
        if len(mouse_points) == 8:
            # change_perspective(frame, mouse_points)
            cv2.destroyWindow('first frame')
            frames = 0
    else:
        # skip 20 frames to have output close to realtime, change to 1 if you run on GPU
        while skip % 20 != 0:
            ret, frame = cap.read()
            skip += 1

        if ret:
            skip += 1
            img = prep_image(frame, inp_dim)
            #        cv2.imshow("a", frame)
            im_dim = frame.shape[1], frame.shape[0]
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)

            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            with torch.no_grad():
                output = model(Variable(img, volatile=True), CUDA)
            output = write_results(output, confidence, num_classes, nms_conf=nms_thesh)

            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.4f}".format(frames / (time.time() - start)))
                cv2.imshow("frame", frame)
                cv2.imwrite('./Outputs/frame' + str(frames) + '.jpg', frame)
                # output_video.write(frame)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue

            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(416 / im_dim, 1)[0].view(-1, 1)

            output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
            output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

            output[:, 1:5] /= scaling_factor

            for i in range(output.shape[0]):
                output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

            # compute_euclidean_distance(output, frame)
            person_flags = create_birds_eye_view(frame, mouse_points, output, bird_view_video)

            classes = load_classes('coco.names')
            colors = pkl.load(open("pallete", "rb"))

            # list(map(lambda x: write(x, frame), output))
            for i in range(len(output)):
                write(output[i], frame)
            # output_video.write(frame)
            cv2.imwrite('./Outputs/frame' + str(frames) + '.jpg', frame)
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print(time.time() - start)
            print("FPS of the video is {:5.2f}".format(frames / (time.time() - start)))
        else:
            break

bird_view_video.release()
# output_video.release()
img_array = []

img_array = []
for i in range(2, frames - 1):
    img = cv2.imread('./Outputs/frame'+str(i)+'.jpg')
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('marked_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 14, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()




