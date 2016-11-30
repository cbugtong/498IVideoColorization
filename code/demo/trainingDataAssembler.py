import numpy as np
import matplotlib.pyplot as plt
import caffe
import os
import skimage.color as color
import scipy.ndimage.interpolation as sni
import csv
import operator

sceneClassTrain = open('sceneClassTrain.txt','w')

gpu_id = 0
caffe.set_mode_gpu()
caffe.set_device(gpu_id)
net = caffe.Net('../placesCNN/places205CNN_deploy.prototxt', 1, weights='../placesCNN/places205CNN_iter_300000.caffemodel')

scenesDir = './Scenes/'
classes = ['beach','forest','house','river','roads','sky','snow','urban']

for i in range(0,len(classes)):
	for subdir, dirs, files in os.walk(scenesDir+classes[i]):
		for xfile in files:
			if(xfile.endswith('.png') or xfile.endswith('.jpg')):
				imgfilename = os.path.join(subdir,xfile)
				(H_in,W_in) = net.blobs['data'].data.shape[2:] # get input shape

				img_rgb = caffe.io.load_image(imgfilename)
				img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
				img_l = img_lab[:,:,0] # pull out L channel
				(H_orig,W_orig) = img_rgb.shape[:2] # original image size

				img_lab_bw = img_lab.copy()
				img_lab_bw[:,:,1:] = 0
				img_rgb_bw = color.lab2rgb(img_lab_bw)

				img_rs = caffe.io.resize_image(img_rgb,(H_in,W_in)) # resize image to network input size
				img_lab_rs = color.rgb2lab(img_rs)
				img_l_rs = img_lab_rs[:,:,0]

				net.blobs['data'].data[0,0,:,:] = img_l_rs-50 # subtract 50 for mean-centering
				net.forward()

				probs = net.blobs['prob'].data[0]

				# throw the probs and the label in a file
				sceneClassTrain.write(str(i)+', '+str(probs).replace('\n','')+'\n')
	print 'Done with '+classes[i]
