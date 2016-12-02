# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import numpy as np
import matplotlib.pyplot as plt
import skimage.color as color
import scipy.ndimage.interpolation as sni
import csv
import operator

np.set_printoptions(threshold=np.nan)

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

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=10,kernel='rbf',probability=True)

# We learn the image probability distribution
classifier.fit(np.array(trainX),np.array(trainY))

classes = ['beach','forest','house','river','roads','sky','snow','urban']

Prob_test=classifier.predict_proba(testX)
loss=0
for i in range(len(testY)):
	GT=testY[i]
	Prediction=classifier.predict(testX[i]);
	if not GT==Prediction:
		loss+=1
		# print 'wrong: '+str(i)
		# print GT
		# print Prob_test[i,:]
print float(loss)/len(testY)

expected = testY
predicted = classifier.predict(testX)

# print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
# print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

sceneClassTrain.close();