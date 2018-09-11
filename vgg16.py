import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class vgg16:
    """ this class build vgg16 model by calling build_model func 
        args:
            inputs:
                trainingX : training images np array with shape [num of images, img height , img width , num of channels]
                trainingY : training labels np array with shape [num of images, num of labels "one hot encoding"]
                (optional) 
                testX : test images np array with shape [num of images, img height , img width , num of channels]
                testY : test labels  np array with shape [num of images, num of labels "one hot encoding"]
         Architecture :
             ""Hint : conv3-64 mean conv layer with kernal size (3,3) and num of filters = 64""
             
            -> conv3-64
            -> conv3-64  -> maxpool 
            -> conv3-128
            -> conv3-128 -> maxpool
            -> conv3-256
            -> conv3-256
            -> conv1-256 -> maxpool
            -> conv3-512
            -> conv3-512
            -> conv1-512 -> maxpool
            -> conv3-512
            -> conv3-512
            -> conv1-512 -> maxpool
            -> FC100 'relu'
            -> FC50 'relu'
            -> FC10  'softmax'
    """
    def __init__(self,sess,trainingX,trainingY,testX="",testY=""):
        #create place holders for inputs and outputs
        shape = [None,trainingX.shape[1],trainingX.shape[2],trainingX.shape[3]]
        self.trainingX = tf.placeholder(shape = shape,name='trainingX',dtype=tf.float32)
        self.trainingY = tf.placeholder(tf.float32,shape = [None,trainingY.shape[1]],name='trainingY')
        self.numOfTraining = trainingX.shape[0]
        self.orginalTrainingX = trainingX 
        self.orginalTrainingY = trainingY
        self.orginalTestX = testX
        self.orginalTestY = testY
        self.sess = sess
        if testX !="":
            shape = [None,testX.shape[1],testX.shape[2],testX.shape[3]]
            self.testX = tf.placeholder(tf.float32,shape =shape,name='testX')
            self.testY = tf.placeholder(tf.float32,shape = [None,testY.shape[1]],name='testY')
            self.numOfTest = testX.shape[0]        
        #list of tubes contain input parameters for func new_conv for every conv layer in the model, we use it whil bulding model
        # every tube has 3 numbers , one string and boolean   (numOfinputChannels,numOfFilters,filterSize,name of layer ,pool)
        self.conv_parameters = [(3,64,3,'conv1',False),(64,64,3,'conv2',True),(64,128,3,'conv3',False),(128,128,3,'conv4',True),\
                                (128,256,3,'conv5',False),(256,256,3,'conv6',False),(256,256,1,'conv7',True),(256,512,3,'conv8',False),\
                                (512,512,3,'conv9',False),(512,512,1,'conv10',True),(512,512,3,'conv11',False),(512,512,3,'conv12',False),\
                                (512,512,1,'conv13',True)]
    def new_weights(self,shape):
        """
        create tf variable with the given shape
        """
        return tf.Variable(tf.truncated_normal(shape=shape,stddev= 0.05))
    
    def new_biases(self,lenth):
        """
        create tf variable with the given shape
        """
        return tf.Variable(tf.constant(0.5,shape=[lenth]))
    
    def new_conv(self,inputt,
             numOfinputChannels,
             numOfFilters,
             filterSize,
             name,
             pooling=False):
        #shape of our weights "tensorflow doc write it like that"
        shape = [filterSize,filterSize,numOfinputChannels,numOfFilters]
        weights = self.new_weights(shape= shape)
        biases = self.new_biases(lenth = numOfFilters)
        layer = tf.nn.conv2d(inputt,filter=weights,strides=[1,1,1,1],padding='SAME',name=name)
        layer+=biases
        if(pooling):
            name = name+ "_maxPool"
            layer= tf.nn.max_pool(layer,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name=name)
        
        layer = tf.nn.relu(layer)
        
        return layer,weights
    
    def flatten(self,layer):
        """
        convert layer with 4D to 2D --> [numOfImages,all features]
        """
        
        #get layer shape 
        shape = layer.get_shape() # [num of images , height , weidth , num of channels]
        numOfFeatures = np.array(shape[1:4],dtype=int).prod() # == height * weidth * num of channels
        #reshape our layer to be [numOfImages,all features]
        layer = tf.reshape(layer,[-1,numOfFeatures])
        
        return layer,numOfFeatures
    
    def new_fc_layer(self,inputt, #input layer and we assumed that it's shpa will be 2D [num of images , num of inputs]
                     numOfInputs,
                     numOfOutputs,
                     relu=True):
        """ it return fc layer with shape [num of images , num of outputs] """
        #shape of wieghts 
        shape = [numOfInputs,numOfOutputs]
        #create weights and biases 
        weights = self.new_weights(shape= shape)
        biases = self.new_biases(numOfOutputs)
        
        #compute our output layer 
        layer = tf.matmul(inputt,weights) + biases
        if(relu):
            layer = tf.nn.relu(layer)
            
        return layer
        
    def build(self):
        """ build the graph """
        #list where we will append our layers
        layers = []
        weights = []
        #create first conv layer with our trainingX data as input
        numOfinputChannels,numOfFilters,filterSize,name,pool = self.conv_parameters[0]
        layer ,weight= self.new_conv(self.trainingX,numOfinputChannels,numOfFilters,filterSize,name,pool)
        layers.append(layer)
        weights.append(weight)
        #create rest of conv layers
        for conv in self.conv_parameters[1:]:
            numOfinputChannels,numOfFilters,filterSize,name,pool = conv
            layer,weight =  self.new_conv(layer,numOfinputChannels,numOfFilters,filterSize,name,pool)
            layers.append(layer)
            weights.append(weight)
        
        #create FC layers
        # first flatten last conv layer to be 2D
        layer ,numOfFeatures= self.flatten(layer)
        layer = self.new_fc_layer(layer,numOfFeatures,100)
        layers.append(layer)
        layer = self.new_fc_layer(layer,100,50)
        layers.append(layer)
        outLayer = self.new_fc_layer(layer,50,6,False) # last layer 
        
        return outLayer,layers,weights
    
    def train(self,learning_rate=1e-4,numOfIteration=30,batchSize=64):
        outLayer , layers ,weights= self.build()
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=outLayer,labels=self.trainingY)
        cost = tf.reduce_mean(cross_entropy)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        y_pred_cls = tf.argmax(tf.nn.softmax(outLayer), axis=1)
        y_true_cls =  tf.argmax(self.trainingY,axis=1)
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())
        numOfBatch = self.numOfTraining // 64
        print("num of batches: ",numOfBatch)
        for i in range(numOfIteration):
            print("iteration num: ",i)
            for k in range(numOfBatch):
                start = k*64
                end = (k+1)*64
                batchX = self.orginalTrainingX[start:end,:,:,:]
                batchY = self.orginalTrainingY[start:end,:]
                feed_dict_train = {self.trainingX :batchX , self.trainingY : batchY}
                self.sess.run(optimizer, feed_dict=feed_dict_train)
            # to make sure that we get all training data 
            if(self.numOfTraining%numOfBatch!=0):
                start = numOfBatch*64
                end = self.numOfTraining
                batchX = self.orginalTrainingX[start:end,:,:,:]
                batchY = self.orginalTrainingY[start:end,:]
                
            #if i % 10 == 0:
            if i:
                # Calculate the accuracy on the training-set.
                acc = self.sess.run(accuracy, feed_dict=feed_dict_train)
    
                # Message for printing.
                msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
    
                # Print it.
                print(msg.format(i + 1, acc))
        save_path = saver.save(self.sess,"./model.ckpt")

        return outLayer,layers,weights
    
    def test(self,y):
        y_pred_cls = np.zeros(shape=self.numOfTest, dtype=np.int)
        y_pred_cls = tf.argmax(tf.nn.softmax(y), axis=1)
        y_true_cls =  tf.argmax(self.trainingY,axis=1)
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        feed_test = {self.trainingX :self.orginalTestX, self.trainingY:self.orginalTestY}
        y_pred_cls= self.sess.run(y_pred_cls,feed_dict=feed_test)
        acc = self.sess.run(accuracy, feed_dict=feed_test)
        print("Test Accuracy : ",acc)
        return y_pred_cls
    
    def plotExampelsErrors(self,y_pred_cls,y_true_cls):
        errors = (y_pred_cls != y_true_cls)
        images = self.orginalTestX[errors]
        clsPred = y_pred_cls[errors]
        clsTrue = y_true_cls[errors]
        fig , axeslist = plt.subplots(ncols = 3 , nrows=3,figsize=(10,10))
        for ind,i in enumerate(images[0:9,:,:]):
            axeslist.ravel()[ind].imshow(images[ind])
            axeslist.ravel()[ind].set_title("true : {0} , pred : {1}".format(clsTrue[ind],clsPred[ind]))
        
        