import numpy as np
import caffe
import matplotlib.pyplot as plt
from PIL import Image

'''
Applies a weighted average between each pixel in image and each in heatmap

Preconditions:
  - image is an rgb image file opened via caffe
  - heatmap is an rgb image file opened via caffe
'''
def applyAverages(image, heatmap):
  # Get the image dimensions
  imWidth, imHeight = image.shape[:2]

  # Resize the heatmap to match the image dimensions
  heatmapImageResized = caffe.io.resize_image(heatmap, (imWidth, imHeight))

  w = 0.9

  for i in range(0,imWidth):
    for j in range(0,imHeight):
      (i_r, i_g, i_b) = image[i,j];

      (h_r, h_g, h_b) = heatmap[i,j]

      image[i,j] = (i_r*(1-w)+h_r*w, i_g*(1-w)+h_g*w, i_b*(1-w)+h_b*w)

image = caffe.io.load_image('imgs/city.jpg')
heatmap = caffe.io.load_image('heatmaps/heatmap_beach.png')

# average it
applyAverages(image,heatmap)

plt.imshow(image)
plt.show()