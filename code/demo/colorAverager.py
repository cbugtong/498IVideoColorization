import numpy as np
import matplotlib.pyplot as plt
import caffe
import os
from PIL import Image

'''
This script determines the average color/palette (in RGB) for a scene
'''

# Iterate through each image

scenesDir = './Scenes/'
classes = ['beach','forest','house','river','roads','sky','snow','urban']

averageColors = []

(hMapWidth,hMapHeight)=(256,256)

# Iterate over each class
for i in range(0,len(classes)):
	# Initialize the heatmap
	hMap = np.zeros((hMapWidth,hMapHeight,3))
	numPics = 0

	for subdir, dirs, files in os.walk(scenesDir+classes[i]):
		# Iterate over each image file
		for imgfile in files:
			if(imgfile.endswith('.png') or imgfile.endswith('.jpg')):
				imgfilename = os.path.join(subdir,imgfile)

				# Ensure that the images are scaled appropriately
				caffeimg = caffe.io.load_image(imgfilename)
				rs = caffe.io.resize_image(caffeimg,(hMapWidth,hMapHeight))

				# Add to the average
				hMap += rs
				
				numPics += 1
		# Divide by the number of images to get the average
		hMap /= numPics

		# Show the heatmap
		# plt.imshow(hMap)
		# plt.savefig('heatmap_'+str(i))

		# Save the heatmap
		hMapSaveCopy = Image.new('RGB', (hMapWidth,hMapHeight))
		hMapSaveCopyPixels = hMapSaveCopy.load()

		for x in range (0,hMapWidth):
			for y in range (0,hMapHeight):
				hMapSaveCopyPixels[y,x] = (int(hMap[x,y][0]*256),int(hMap[x,y][1]*256),int(hMap[x,y][2]*256))

		hMapSaveCopy.save("heatmaps/heatmap_"+str(i)+".png","PNG")

	print "Done with "+classes[i]