import dataProcessing
import vgg16
import tensorflow as tf
 
trainingPath ='datasets/train_signs.h5'
testPath = 'datasets/test_signs.h5'
dataProcess = dataProcessing.imageDataProcessing(trainingPath=trainingPath,testPath=testPath)
#load data
indexOftrainX = "train_set_x"   #index of training x data in train.h5 file
indexOftrainY = "train_set_y"   #index of training y data in train.h5 file
indexOftestX = "test_set_x"    #index of test x  data in test.h5 file
indexOftestY = "test_set_y"    #index of test y data in test.h5 file
trainingX,trainingY,testX,testY = dataProcess.loadH5Data(indexOftrainX,indexOftrainY,indexOftestX,indexOftestY)

# reshape to be 1D so we can use on_hot function later
testY = testY.reshape(120)
trainingY = trainingY.reshape(1080)





#plot some figures 
#dataProcess.plot_figure(trainingX,trainingY,3,3)


#one hot encoding 
trainingY = dataProcess.on_hot((trainingY))
testY = dataProcess.on_hot((testY))

#print data shapes
print ("number of training examples = " + str(trainingX.shape[0]))
print ("number of test examples = " + str(testX.shape[0]))
print ("X_train shape: " + str(trainingX.shape))
print ("Y_train shape: " + str(trainingY.shape))
print ("X_test shape: " + str(testX.shape))
print ("Y_test shape: " + str(testY.shape))

#saver = tf.train.Saver()

with tf.Session() as sess:
    model = vgg16.vgg16(sess,trainingX,trainingY,testX,testY)
    y_true_cls =  tf.argmax(testY,axis=1)
    y_true_cls = sess.run(y_true_cls)
    outlayer,layers,weights= model.train()
    y_pred_cls = model.test(outlayer)
    model.plotExampelsErrors(y_pred_cls,y_true_cls)

'''
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
'''