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

scenesDir = './Scenes/'
classes = ['beach','forest','house','river','roads','sky','snow','urban']

averageColors = []

for i in range(0,len(classes)):
	averageColor = 0

	for subdir, dirs, files in os.walk(scenesDir+classes[i]):
		for xfile in files:
			if(xfile.endswith('.png') or xfile.endswith('.jpg')):
				imgfilename = os.path.join(subdir,xfile)

				# Ensure that it has the right dimensions
				
	print "Done with "+classes[i]