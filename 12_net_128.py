
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import numpy as np
import operator as op
import sys
import datetime
import csv
from keras.models import model_from_json


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

class cnn_keras:

    def __init__(self ,batch_size):

        """
        Constructor
        """
        print("Network object created!")
        self.batches = batch_size


    def load_data(self ,file_name ,row ,col ,channels,samples):

        """
        This function takes the name of the file and 
        creates the feature array and one hot vector of that file
        """

        fr = open(file_name ,"r")
        csr = csv.reader(fr)
        one_hot = np.zeros((samples,2))
        feature_arr = []

        ct = 0
        for temp in csr:  # for each line / one sample j=0
            onesample=np.zeros([row,col,channels]) # dimensions of row, col and channel respectively.
            total=len(temp)
           
            one_hot[ct][int(temp[-1])] = 1            
            print("Reading line ",ct, " Length of the line is ",total)
            ct += 1  # keeps track of the number of images
            j = 0
            
            while(j<total-3):  # Since the last entry is the label
                r=0 
                while(r<row):
                    c=0
                    while(c<col):
                        onesample[r][c][0]=float(int(temp[j])/255)  # each pixel is getting normalised
                        onesample[r][c][1]=float(int(temp[j+1])/255)
                        onesample[r][c][2]=float(int(temp[j+2])/255)
                        j = j+3
                        c = c+1
                    r=r+1 
                
            
            feature_arr.append(onesample) # Appends the entire image

        dummy = np.asarray(feature_arr)  # converts the list to a numpy array
        dim = len(feature_arr)   # calculates the dimensions of the whole feature array
        feature_arr = np.reshape(dummy, (dim, row, col, channels))   # converting the list to a 4D numpy array  # self.feature_arr_test_new = np.transpose(self.feature_arr_test_new, (0, 3, 1, 2))  # use in case of Theano
        print("Shape of feature array is ",feature_arr.shape)


        return(feature_arr,one_hot)
    
    def bounding_box_coordinates(self,file_name): 
        """
        Inputs the name of the file and creates a matrix of size 952 X 4 where 4 is the co-ordinates of the
        boundary points
        """
        fr = open(file_name,"r")
        csr = csv.reader(fr)

        boundaries = np.zeros(shape=(952,4)) # stores the co-ordinates in the order start_X,start_Y,end_X,end_Y
        orientation = [] # Stores the orientation of the face i.e. frontal, lateral, semi lateral bla bla..
        itr = 0

        for temp in csr:
            boundaries[itr][0] = temp[-5]
            boundaries[itr][1] = temp[-4]
            boundaries[itr][2] = temp[-3]
            #boundaries[itr][3] = temp[-2]
            #orientation.append = temp[-1]
            itr += 1

        print("Shape of boundaries vector is ",boundaries.shape)
        return(boundaries,orientation)
    
    def create_network(self): 
        """
        Creates the basic layout of the network. Creates the number of layers
        number of nodes in each layer and specifies the activation functions

        """
        print("Inside Create Network")
        self.model = Sequential()       # function to sequentially add layers

        # creates model for CALTECH_101
        self.model.add(Convolution2D(6,5,5,subsample=(1,1), border_mode='valid',activation='sigmoid',input_shape = (128,128,3)))
        print( "**** First convolution layer complete ****")
        self.model.add(MaxPooling2D(pool_size=(2,2)))
        print( "**** First Maxpool layer complete ****")
        self.model.add(Convolution2D(8,3,3,subsample=(1,1), border_mode='valid',activation='sigmoid'))
        print("**** Second convolution layer complete ****")
        self.model.add(MaxPooling2D(pool_size=(1,1)))
        print("**** Second Maxpool layer complete ****")
        self.model.add(Convolution2D(10,3,3,subsample=(1,1), border_mode='valid',activation='sigmoid'))
        print("**** Third convolution layer complete ****")
        self.model.add(MaxPooling2D( pool_size= (1,1)))
        print("**** Third Maxpool layer complete ****")
        self.model.add(Flatten())
        print("**** Image Flattened ****")
        self.model.add(Dense(2, init='uniform', activation='sigmoid'))
        print("**** Dense layer added ****")


        # Compile the model
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()


    def train_network(self,no_epoch):

        """
        Inputs number of epochs and trains the system accordingly

        """
        # Fit the model

    # Train_network(model,X,Y)

        # writes the summary of the network in this file
        file_manip = open("Network_summary_128_net.txt","a")

        file_manip.write("\n Dimension of the train matrix ")
        file_manip.write(str(self.feature_arr_train.shape))
        file_manip.write( "\n Dimension of the test matrix ")
        file_manip.write(str(self.feature_arr_test.shape))


        now = datetime.datetime.now()

        file_manip.write("\n Chronicles : ")
        file_manip.write(str(now))
        file_manip.write("\n")
        file_manip.write(

        "\n*********   Printing the accuracy measure of train    **********\nTrain score ")
        file_manip.close()

        best_accuracy = 0.0

        for itr in range(0,no_epoch):
            file_manip = open("Network_summary_128_net.txt","a")
            file_manip.write("----------------------\nIn iteration number ")
            file_manip.write(str(itr + 1))
            print( "----------------------\nIn iteration number ",itr+1)

            # training is being done iteratively
            self.accuracy_measures = self.model.fit(self.feature_arr_train, self.one_hot_train, epochs = 1, batch_size=self.batches, validation_data = (self.feature_arr_test, self.one_hot_test), verbose = 1)

            dummy = op.itemgetter(0)(self.accuracy_measures.history["val_acc"])
            file_manip.write("\nTrain loss ")
            file_manip.write((str)(self.accuracy_measures.history["val_loss"]))
            file_manip.write("\nTrain accuracy ")
            file_manip.write((str)(self.accuracy_measures.history["val_acc"]))

            if(dummy > best_accuracy):
                best_accuracy = dummy

            file_manip.close()
            self.save_model()  # saves the model after each epoch
            self.test_predict()  # predicts the accuracy of ths system after each epoch


        file_manip = open("Network_summary_128_net.txt","a")
        file_manip.write("\n\nAfter the end of training phase best found accuracy is ")
        file_manip.write(str(best_accuracy)) 
        file_manip.close()


    def accuracy(self):

        """
        Measures the accuracy of the trained system

        """

        file_manip = open("Network_summary_128_net.txt","a")
        scores = self.model.evaluate(self.feature_arr_test, self.one_hot_test, batch_size = self.batches,verbose = 1)
        file_manip.write( "\nFrom 'Evaluate' function test %s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))
        print("\nFrom 'Evaluate' function test %s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))

        file_manip.close()

    
    def create_confusion_matrix(self):

        """
        From the one hot predicted matrix and the one hot test matrix creates the 
        confusion matrix
        """
        confusion_mat = np.zeros(shape=(self.unique_word_count,self.unique_word_count))

        predict_matrix_len = len(self.one_hot_predicted)        
        for i in range(0,predict_matrix_len):
            flag = 0            
            for j in range(0,self.unique_word_count):  # if the prediction is wrong
                if(self.one_hot_predicted[i][j] != self.one_hot_test[i][j]):
                    temp1 = np.argmax(self.one_hot_test[i])                    
                    temp2 = np.argmax(self.one_hot_predicted[i])
                    confusion_mat[temp2][temp1] += 1
                    flag = 1
            if(flag == 0):  # if the prediction is correct
                    temp = np.argmax(self.one_hot_test[i])
                    confusion_mat[temp][temp] += 1    

        for i in range(0,self.unique_word_count):
            for j in range(0,self.unique_word_count):
                if(i != j):
                    confusion_mat[i][j] /= 2  # calculates the wrongly predicted samples 


        print("\nTRUE CLASSES->")   
        print("PREDICTED CLASSES\n |\n v")
        print(confusion_mat)             
    

    def save_model(self):
        """
        Saves the present state of the trained network
        we can either start training from scratch or build on the previous training
        """
        # write as JSON and saves the model
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model_128_net.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model_128_net.h5")
        # print("Saved model to disk")


    def load_model(self):
        """
        Loads the model and builds on it        
        """

        # loads the model
        json_file = open('model_128_net.json', 'r')
        self.model = json_file.read()
        json_file.close()
        self.model = model_from_json(self.model)
        # load weights into new model
        self.model.load_weights("model_128_net.h5")
        print("Loaded model from disk")

        # evaluate loaded model on test data
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    
    def test_predict(self):
        """
        predicts the classes of the test set
        """
        test_size = len(self.feature_arr_test)    
        self.unique_word_count = 2 # hardcoded for binary classification face/non-face            

        classes = self.model.predict(self.feature_arr_test,batch_size = self.batches)  # gives a 2d array of probability of likelihood of a label
        self.one_hot_predicted = np.zeros(shape=(test_size,self.unique_word_count))

        # creates the predicted one hot vector based on the max probability of classes
        for i in range(0,test_size):
            maximum = classes[i][0]
            idx = 0            
            for j in range(0,self.unique_word_count):
                if(classes[i][j] > maximum):
                    maximum = classes[i][j]
                    idx = j
            self.one_hot_predicted[i][idx] = 1 # the object is assigned to the class with maximum probability 

        # calculates the correct and wrong predictions
        self.wrong_prediction_ct = 0

        for i in range(0,test_size):
            for j in range(0,self.unique_word_count):
                if(self.one_hot_predicted[i][j] != self.one_hot_test[i][j]):
                    self.wrong_prediction_ct += 1    

        self.wrong_prediction_ct /= 2   # calculates the wrongly predicted samples 
        self.correct_prediction_ct = test_size - self.wrong_prediction_ct    # calculates the correctly predicted samples   


        file_manip = open("Network_summary_128_net.txt","a")

        self.correct_prediction_ct = self.correct_prediction_ct / test_size 
        file_manip.write("\nMeasured accuracy ")
        file_manip.write(str(self.correct_prediction_ct))  # prints the accuracy in the file

        print("\nMeasured accuracy ",self.correct_prediction_ct)
        file_manip.close() 

    
    def train_array_creation(self,x_train,y_train):
        """
        Initializes the feature and one hot vectors
        """
        self.feature_arr_train = x_train        
        self.one_hot_train = y_train
        

    def test_array_creation(self,x_test,y_test):
        """
        Initializes the test array
        """
    
        self.feature_arr_test = x_test
        self.one_hot_test = y_test


    
    '''
    def predict_single_sample(self):

    '''


def main():

    # iris_train_file_name = "Iris_train.txt"   #input("Enter the training file name ")
    # iris_test_file_name = "Iris_test.txt"   #input("Enter test file name ")


    train_file_name = "128_train.csv"  # input("Enter the MNIST training file name ")
    test_file_name = "128_test.csv"  # input("Enter MNIST test file name ")

    epochs = 50  # input("Enter how many epochs ")
    batch_size = 32  # input("Enter batch size ")

    obj = cnn_keras(batch_size)  # obj.load_iris_data(iris_test_file_name,iris_train_file_name)    # initial matrix created

    row = 128
    col = 128
    channels = 3
    
    
    arg = sys.argv[1]
    
    if (arg == "new"):
       print("***********************  Reading Training Data **************************")
       (x1,y1) = obj.load_data(train_file_name,row,col,channels,20008)    # initial matrix created
       print( "\n********* Training file reading complete!!**********\n")
    
    
       print("*********************** Reading Test Data **************************")
       (x2,y2) = obj.load_data(test_file_name,row,col,channels,11380)
       print("\n********* Test file reading complete!!**********\n")
    
       print("*************** PRINTING DIMENSIONS  ****************")
       print("train feature matrix is having shape ",x1.shape)
       print("test feature matrix is having shape ",x2.shape)    
       print("train one hot matrix is having shape ",y1.shape)        
       print("test one hot matrix is having shape ",y2.shape)    
       print("***********************************************")
    
       obj.train_array_creation(x1,y1)
       obj.test_array_creation(x2,y2)
       obj.create_network()               # Network created
       obj.train_network (epochs)       # Trains the network
    
    else:
        
        test_file_name = "128_test.csv"
        print("*********************** Reading Test Data **************************")
        (x2,y2) = obj.load_data(test_file_name,row,col,channels,11380)
        print("\n********* Test file reading complete!!**********\n")
        print("*************** PRINTING DIMENSIONS  ****************")
        print("test feature matrix is having shape ",x2.shape)
        print("test one hot matrix is having shape ",y2.shape)
        
        obj.test_array_creation(x2,y2)
        obj.load_model()  # loads the previously trained model
        obj.accuracy()  # measures the accuracy of the trained system    # obj.new_test()  # loads the new test file and measures the accuracy of it
        obj.test_predict() # predicts the classes of the test batch
        #obj.create_confusion_matrix()  # creates and prints the confusion matrix   obj.save_model() # saves the model as "model"


main()


