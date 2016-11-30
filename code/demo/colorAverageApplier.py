import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.color as color
import scipy.ndimage.interpolation as sni
import csv
import operator
import Image

'''
This script determines the average color/palette (in RGB) for a scene
'''

# Iterate through each image

def applyAverages(image, average_colors):
  average_width, average_height = average_colors.size
  width, height = image.size
  multiplier_x = average_width / (float) width
  multiplier_y = average_height / (float) height
  rgb_im = image.convert('RGB')

  for i in range(0,width):
    for j in range(0,height):
      r,g,b = rgb_im.getPixel(i,j);

      average_x_index=(int) (multiplier_x * i)
      average_y_index=(int) (multiplier_y * j)

      average_r, average_g, average_b = average_colors[average_x_index][average_y_index]

      new_color=((r * .9 + average_r * .1), (.9 + average_g * .1), (b * .9 + average_b * .1))

      image.putpixel( (i,j), new_color)

  return image