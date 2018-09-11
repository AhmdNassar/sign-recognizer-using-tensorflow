import h5py
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import random
import matplotlib.pyplot as plt

class imageDataProcessing():
    """
    optimze calss for image data processing 
    inputs :
        - image width : int number if image less than it will padding with zeros to match given width
        - image height : int number if image less than it will padding with zeros to match given height
        - training data path : if training data split into x and y pass both in tuble of string (xPath , yPath)
        - test data (option) : if test data split into x and y pass both in tuble of string (xPath , yPath)
    
    functions: 
        include five functions (every functions has its own documentation inside it)
        
 (Done) - loadH5Data :  load data of type .h5
        - splitData  : split data into training and test data and validation option is available
 (Done) - onHot      : encoding labels into one hot matrix 
        - plot_figure: plot some figures of our data 
        - shuffle    : shuffle data

    # TODO : add more load data function for other types
             add function to check Nan and convert it to zeros or any thing else 
             
    """
    def __init__(self,trainingPath,imgWidth=100,imgHeight=100,testPath="none"):
        self.imgWidth = imgWidth # we use it when data not h5 file 
        self.imgHeight = imgHeight # we use it when data not h5 file 
        self.trainingPath = str(trainingPath) # to be sure that it's string 
        self.testPath=str(testPath) # to be sure that it's string 
        
    def loadH5Data(self,XtrainingIndex,YtrainingIndex,XtestIndex="none",YtestIndex="none"):
        """
        load training data and if test data path given load it also 
        args:
            input:
                XtrainingIndex : index of X training data into training h5 file 
                YtrainingIndex : index of Y training data into training  h5 file 
                
                (option):    XtestIndex : index of X test data into test h5 file 
                (option):    YtestIndex : index of X test data into test h5 file 
                
        returns:
            trainingData : tuble(x,y)
            test         : tuble (x,y) if available 
        """
        training = h5py.File(self.trainingPath,'r')
        trainingX = np.array(training[XtrainingIndex][:])  
        trainingY = np.array(training[YtrainingIndex][:])
        trainingY = trainingY.reshape((1,trainingY.shape[0])) #reshape labels to be (1,m) where m = num of training examples 
        
        if(self.testPath!="none"): # check if test path given to load test data and return training and test data 
            assert (XtestIndex!='none'),"x test index not given!"
            assert (XtestIndex!='none'),"y test index not given!"
            test = h5py.File(self.testPath,'r')
            testX = np.array(test[XtestIndex][:])
            testY = np.array(test[YtestIndex][:])
            testY=testY.reshape((1,testY.shape[0])) #reshape labels to be (1,m) where m = num of test examples 
            return trainingX , trainingY , testX , testY
        
        return trainingX , trainingY # return training data only if test path not given 
    
    def on_hot(self,labels):
        """ 
        convert labels into one hot coding 
        
        args:
            - input : list of labels 
            - output : np array of one hot encoding 
        
        """
        encoder = LabelBinarizer()
        encodedLabels = encoder.fit_transform(labels)
        return encodedLabels


    def plot_figure(self,data,labels,ncols = 1 ,nrows =1,fig_width = 10 ,fig_height = 10 ):
        """
        plot figures of our data, where num of fogures = col*row
        
        args:
            input:
                - data which will plot figures from it
                - labels of data
                - ncols :number of columns of subplots wanted in the display
                - nrows : number of rows of subplots wanted in the figure
                - nimages : number of all images in data set
                - fig_width : width for each image we will plot 
                - fig_height : height for each image we will plot
        """
        m = data.shape[0]
        # list of random indexs for images that we will plot 
        random_index = random.sample(range(0,m),nrows*ncols)
        fig , axeslist = plt.subplots(ncols = ncols , nrows=nrows,figsize=(fig_width,fig_height))
        for ind,i in enumerate(random_index):
            axeslist.ravel()[ind].imshow(data[i])
            axeslist.ravel()[ind].set_title(labels[i])
            
            
        
        