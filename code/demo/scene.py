import numpy as np
import matplotlib.pyplot as plt
import caffe
import os
import skimage.color as color
import scipy.ndimage.interpolation as sni
import csv
import operator

'''
This is a script that tests out places205CNN
'''

gpu_id = 0
caffe.set_mode_gpu()
caffe.set_device(gpu_id)
net = caffe.Net('../placesCNN/places205CNN_deploy.prototxt', 1, weights='../placesCNN/places205CNN_iter_300000.caffemodel')

(H_in,W_in) = net.blobs['data'].data.shape[2:] # get input shape
print net.blobs['data'].data.shape[2:]

img_rgb = caffe.io.load_image('./imgs/sky.jpg')
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

scenes = []
with open('../placesCNN/categoryIndex_places205.csv', 'rb') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
	for row in spamreader:
		scenes.append(', '.join(row))

argMax = np.argmax(net.blobs['prob'].data[0])

probs = net.blobs['prob'].data[0]
dict_index_prob={}
for i in range(0,len(probs)):
	dict_index_prob[scenes[i]]=probs[i]
sorted_x = sorted(dict_index_prob.items(), key=operator.itemgetter(1),reverse=True)

print 'Highest probable scene: '+scenes[argMax]
print 'Highest probability: '+str(probs[argMax])
print sorted_x