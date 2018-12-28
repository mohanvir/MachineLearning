"""
In order to run this file, please make sure the following libraries are installed:
    1) numpy
    2) matplotlib.pyplot
    3) pandas
    4) sklearn
    5) keras

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#------------------ --------------------DATA PREPROCESSING -------------------------------------------------------

# Importing the dataset and setting up independent and dependent variables -------------------
dataset = pd.read_csv('Churn_Modelling.csv') #insert csv file here
X = dataset.iloc[:, 3: 13].values  # Change these to all independent rows, in python the upper bound is excluded
y = dataset.iloc[:, 13].values #insert the independent variable, this should be one single variable.

#change all categorical data into numerical  values  --------------------------------------------
    # this needs to be done to all independent variables that are not in number form!
labelencoder_X_1 = LabelEncoder() #LabelEncoder() is from sklearn.preprocessing
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

#example in dataset provided, encoding male and female categorical to binary
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])


#onehotencoder arre used to make deal with dummy variables that could have been caused.
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:] # for our example data , 3 countries made into dummy variables means we need to remove one, dummy variable effect

#Splitting the Dataset into training and testing.  good ratio is 80 training and 20 testing
from sklearn.model_selection import train_test_split # this also scales
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling there is alot of computation, this eases the computations
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) # we are scaling all of the X_train to fit
X_test = sc.transform(X_test) # we want to keep the same fit scale as above, so we dont call it again ,transform already knows the fit scale

#-----------------------------------Building an ANN----------------------------------------
import keras # this is using TensorFlow backend
from keras.models import Sequential  #this is to initalize nerual netowrks
from keras.layers import Dense # this is how we make the layers
from keras.layers import Dropout # this is for over training
#Dropout is applied to some layers and will this stop overfitting

#Initialize the ANN
classifier = Sequential() # this is the intialization for ANN

#this is adding the input layer and first hiddden layer
classifier.add(Dense(output_dim =6, init = 'uniform', activation ='relu' ,input_dim =11)) #rectifier function
classifier.add(Dropout(rate = 0.1))   # start with rate = 0.1 and go up to 1, closer to 1 there is underfitting
#this is to addd the second hidden layer
classifier.add(Dense(output_dim =6, init = 'uniform', activation ='relu'))
classifier.add(Dropout(rate = 0.1))   # start with rate = 0.1 and go up to 1, closer to 1 there is underfitting

#this is adding the output layer
classifier.add(Dense(output_dim =1, init = 'uniform', activation ='sigmoid')) #this is sigmoid function
#compile
classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics = ['accuracy'])

# Fitting classifier to the Training set
classifier.fit(X_train, y_train,batch_size = 10, nb_epoch=100)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_pred = (y_pred >0.5) #sigmoid function returns probability so we need to make anything higher than 0.5 equal 1

new_prediction = classifier.predict(sc.transform(np.array([[0,0,600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction >0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
