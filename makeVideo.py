import cv2
import numpy as np
import glob

img_array = []

img_array = []
for i in range(120, 720):
    img = cv2.imread('./Output_vids/frame'+str(i)+'.jpg')
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

out = cv2.VideoWriter('marked_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 14, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
