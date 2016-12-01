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

	j = 0
	for subdir, dirs, files in os.walk(scenesDir+classes[i]):
		# Iterate over each image file
		for imgfile in files:
			if(imgfile.endswith('.png') or imgfile.endswith('.jpg')):
				# if j >= 20:
				# 	break
				imgfilename = os.path.join(subdir,imgfile)

				# Ensure that the images are scaled appropriately
				caffeimg = caffe.io.load_image(imgfilename)
				rs = caffe.io.resize_image(caffeimg,(hMapWidth,hMapHeight))

				# Add to the average
				for x in range(hMapWidth):
					for y in range(hMapHeight):
						hMap[x][y] = (hMap[x][y][0]+rs[x][y][0],hMap[x][y][1]+rs[x][y][1],hMap[x][y][2]+rs[x][y][2])
				
				numPics += 1
				j+=1
		# Divide by the number of images to get the average
		for x in range(hMapWidth):
			for y in range(hMapHeight):
				hMap[x][y] = (hMap[x][y][0]/numPics,hMap[x][y][1]/numPics,hMap[x][y][2]/numPics)

		# Show the heatmap
		# plt.imshow(hMap)
		# plt.show()

		# Save the heatmap
		j = Image.fromarray(hMap)
		j.save("~/Desktop/heatmap_"+str(i),".jpg")

	print "Done with "+classes[i]