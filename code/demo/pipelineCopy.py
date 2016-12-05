from PIL import Image
import caffe

import numpy as np
import skimage.color as color
import scipy.ndimage.interpolation as sni
from sklearn import svm
import matplotlib.pyplot as plt

'''
The full colorization pipeline
'''
scenes = ['beach','forest','house','river','roads','sky','snow','urban']

'''
Given an imageFileName, this method converts the image to black and white and runs it through the richard zhang CNN
Outputs 2D colorized image data
'''
def colorize(imageFileName):
	gpu_id = 0
	caffe.set_mode_gpu()
	caffe.set_device(gpu_id)
	net = caffe.Net('../models/colorization_deploy_v1.prototxt', '../models/colorization_release_v1.caffemodel', caffe.TEST)

	(H_in,W_in) = net.blobs['data_l'].data.shape[2:] # get input shape
	(H_out,W_out) = net.blobs['class8_ab'].data.shape[2:] # get output shape
	net.blobs['Trecip'].data[...] = 6/np.log(10) # 1/T, set annealing temperature
	# (We found that we had introduced a factor of log(10). We will update the arXiv shortly.)

	# load the original image
	img_rgb = caffe.io.load_image(imageFileName)

	img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
	img_l = img_lab[:,:,0] # pull out L channel
	(H_orig,W_orig) = img_rgb.shape[:2] # original image size

	# create grayscale version of image (just for displaying)
	img_lab_bw = img_lab.copy()
	img_lab_bw[:,:,1:] = 0
	img_rgb_bw = color.lab2rgb(img_lab_bw)

	# resize image to network input size
	img_rs = caffe.io.resize_image(img_rgb,(H_in,W_in)) # resize image to network input size
	img_lab_rs = color.rgb2lab(img_rs)
	img_l_rs = img_lab_rs[:,:,0]

	net.blobs['data_l'].data[0,0,:,:] = img_l_rs-50 # subtract 50 for mean-centering
	net.forward() # run network

	ab_dec = net.blobs['class8_ab'].data[0,:,:,:].transpose((1,2,0)) # this is our result
	ab_dec_us = sni.zoom(ab_dec,(1.*H_orig/H_out,1.*W_orig/W_out,1)) # upsample to match size of original image L
	img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
	img_rgb_out = np.clip(color.lab2rgb(img_lab_out),0,1) # convert back to rgb

	return img_rgb_out

'''
Given an imageFileName, this method runs the black and white image through the places205CNN network
'''
def placesCNNRun(imageFileName):
	gpu_id = 0
	caffe.set_mode_gpu()
	caffe.set_device(gpu_id)
	placesNet = caffe.Net('../placesCNN/places205CNN_deploy.prototxt', 1, weights='../placesCNN/places205CNN_iter_300000.caffemodel')

	(H_in,W_in) = placesNet.blobs['data'].data.shape[2:] # get input shape

	img_rgb = caffe.io.load_image(imageFileName)
	img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
	img_l = img_lab[:,:,0] # pull out L channel
	(H_orig,W_orig) = img_rgb.shape[:2] # original image size

	img_lab_bw = img_lab.copy()
	img_lab_bw[:,:,1:] = 0
	img_rgb_bw = color.lab2rgb(img_lab_bw)

	img_rs = caffe.io.resize_image(img_rgb,(H_in,W_in)) # resize image to network input size
	img_lab_rs = color.rgb2lab(img_rs)
	img_l_rs = img_lab_rs[:,:,0]

	placesNet.blobs['data'].data[0,0,:,:] = img_l_rs-50 # subtract 50 for mean-centering
	placesNet.forward()

	return placesNet.blobs['prob'].data[0]

'''
Given an svm, the method opens up the training data file (sceneClassTrain.txt) and fits the svm to the data
'''
def trainSVM(classifier):
	sceneClassTrain = open('sceneClassTrain.txt','r')

	# Assemble the distribution
	trainX = []
	trainY = []
	testX = []
	testY = []
	toggle = False
	for example in sceneClassTrain:
		if toggle:
			trainX.append([float(i) for i in example[3:].rstrip('\n').rstrip(']').split()[1:]])
			trainY.append(int(example[0]))
		else:
			testX.append([float(i) for i in example[3:].rstrip('\n').rstrip(']').split()[1:]])
			testY.append(int(example[0]))
		toggle = not toggle

	n_samples = len(trainY)

	# We learn the image probability distribution
	classifier.fit(np.array(trainX),np.array(trainY))

'''
Given an image and a heatmap, this method returns a weighted average of both
'''
def applyWeightedAverage(image, heatmap, w, threshold):
	# TODO normalize the image scale

	# Get the image dimensions
	(imWidth, imHeight) = image.shape[:2]

	# Initialize the results
	result = np.zeros((imWidth,imHeight,3))

	# Resize the heatmap to match the image dimensions
	heatmap_rs = caffe.io.resize_image(heatmap, (imWidth, imHeight))

	# Convert both the image and the heatmap to lab
	image_lab = color.rgb2lab(image)
	heatmap_rs_lab = color.rgb2lab(heatmap_rs)

	# Compute the weighted average
	# result = image_lab*(1-w)+heatmap_rs_lab*w

	# Code for rgb
	# for i in range(imWidth):
	# 	for j in range(imHeight):
	# 		(i_r,i_g,i_b) = image[i,j]
	# 		(h_r,h_g,h_b) = heatmap_rs[i,j]

	# 		rDiff = i_r - h_r
	# 		gDiff = i_g - h_g
	# 		bDiff = i_b - h_b
	# 		if rDiff > threshold or gDiff > threshold or bDiff > threshold:
	# 			result[i,j] = (i_r*(1-w)+h_r*w,i_g*(1-w)+h_g*w,i_b*(1-w)+h_b*w)

	# 			# Debug: Highlights the affected areas in purple
	# 			# result[i,j] = (1,0,1)
	# 		else:
	# 			result[i,j] = image[i,j]

	# Code for lab
	for i in range(imWidth):
		for j in range(imHeight):
			(i_l,i_a,i_b) = image_lab[i,j]
			(h_l,h_a,h_b) = heatmap_rs_lab[i,j]

			aDiff = i_a - h_a
			bDiff = i_b - h_b
			if aDiff > threshold or bDiff > threshold:
				result[i,j] = (i_l*(1-w)+h_l*w,i_a*(1-w)+h_a*w,i_b*(1-w)+h_b*w)

				# Debug: Highlights the affected areas in purple
				# result[i,j] = (1,0,1)
			else:
				result[i,j] = image_lab[i,j]

	# Convert back to rgb
	result_rgb = color.lab2rgb(result)

	return result_rgb

'''
Given the ground truth and the prediction image, computes the MSE
'''
def getMeanSquareError(gTruth, prediction):
	# Get the image dimensions
	(imWidth, imHeight) = gTruth.shape[:2]

	error = gTruth - prediction

	mse = np.sum(np.sum(error * error))	# TODO: Make sure that this is right

	mse /= imWidth * imHeight

	return mse

'''
The script starts here!
'''
imageFileName = 'imgs/test/suicideforest.jpg'

groundTruth = caffe.io.load_image(imageFileName)

# Put the black and white image through the Richard Zhang pipeline
colorizedImage = colorize(imageFileName)

# Train the scene classifier
classifier = svm.SVC(gamma=10,kernel='rbf',probability=True)
trainSVM(classifier)

# Classify the image:
# 	run through places205CNN
#	give the places205CNN probs to the svm
placesProbs = placesCNNRun(imageFileName)
scenePredictionIndex = classifier.predict(placesProbs)

# Grab the corresponding heatmap
heatmap = caffe.io.load_image('heatmaps/heatmap_'+scenes[scenePredictionIndex]+'.png')

# Show the final results for these parameters
params = [(0.5,0.5), (0.4, 0.4), (0.3, 0.3), (0.2, 0.2), (0.1, 0.1)]
result = []

# Side length
sideLength = np.ceil(np.sqrt(2+len(params)))

fig = plt.figure()
a=fig.add_subplot(sideLength,sideLength,1)
imgplot=plt.imshow(colorizedImage)
a.set_title('Colorized Image')

print "RichZhang MSE: "+str(getMeanSquareError(groundTruth, colorizedImage))

a=fig.add_subplot(sideLength,sideLength,2)
imgplot=plt.imshow(groundTruth)
a.set_title('Ground Truth')

for j in range(len(params)):
	paramTuple = params[j]

	# Based on the loss or mean squared error, apply a weighted average between the heatmap and the loss region
	result.append(applyWeightedAverage(colorizedImage, heatmap, paramTuple[0], paramTuple[1]))

	# Add the weighted average with the paramTuple parameters to the matlab plot
	a=fig.add_subplot(sideLength,sideLength,3+j)
	imgplot=plt.imshow(result[j])
	a.set_title(scenes[scenePredictionIndex]+':('+str(paramTuple[0])+' w, '+str(paramTuple[1])+' t)')

	print "Result #"+str(j+1)+" MSE: "+str(getMeanSquareError(groundTruth,result[j]))

plt.show()

# Consider using Lab so the brightness is maintained
# Make it so you selectively average the heatmap with the colorized image if they differ greater than a certain threshold
# The scale of the image when you're applying the heatmap. Consider normalizing the image to match the heatmap scale
# Calculate MSE