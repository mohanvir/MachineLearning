#trying to predict a stock trend using LSTM. using 5 years of stock prices

#-------------------Data Preprocessing-----------------------------------------
import numpy as np #make arrays from data arrays
import matplotlib.pyplot as plt #visualize results
import pandas as pd #import dataset

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2] # if we only take one we will get an array, but we need a numpy array

#feature scaling standardization vs normalisation   if sigmoid function then use normalization
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1) ) #  we scaled everything to 0 and 1 which will help the normalization function
training_set_scaled = sc.fit_transform(training_set)

#creating a data structure with 60timesteps and 1 output, look at 60 previous timesteps to predict future
X_train = []
y_train = []    # two empty lists

for i in range(60, 1258): # last index of the list is 1257 but it is excluded from the list, 60 is how many back steps
    X_train.append(training_set_scaled[i-60 : i, 0]) #memorize the last 60 to predict the next one at each index of the list we will have 60 elements, we start at the 61st day
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshapping number of predictors we can use, or indicators can add more indicators here
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#---------------- Building the RNN ----------- with drop regulation
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential() # this is gonna be a graph
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = ( X_train.shape[1], 1)))
#number units,
#return sequence = true, because this is stacked and we will have multiple layers default is false
#input shape  This is the 3d, observations, timesteps and indicators, we only need to add the last two
regressor.add(Dropout(0.2))
#this is to regularlize, good estimate is to drop 20%

#adding 3 more LSTM layers with Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50)) #return_sequences = false because this is the output layer
regressor.add(Dropout(0.2))

#adding output layer
regressor.add(Dense(units =1)) #how many dimensions in output layer this is time +1

#------------ Making the predicitions and visualising the results
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') #this is from sequential class, we could try RMSprop but it was less effective for me


#---Joining the training set and the RNN
regressor.fit(X_train, y_train, epochs= 100, batch_size = 32)  # this will connect and run
#fit will connect RNN and data

#-----------------Making the predictions and graphing-----------------------------------------------
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#getting predicted prices, we need 60 previous days to predict every next day, need both datasets for jan 5thwe need to have jan 4st plus last 60 days
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0) #open is the name of the column
inputs =  dataset_total[len(dataset_total) - len(dataset_test) -60 : ].values       # need 60 previous days, and make it into numpy array
inputs = inputs.reshape(-1,+1) #getsrid of warning of numpy dimensions
inputs = sc.transform(inputs)


X_test = [] #  empty list for tests

for i in range(60, 80): # last index of the list is 1257 but it is excluded from the list, 60 is how many back steps
    X_test.append(inputs[i-60 : i, 0]) #memorize the last 60 to predict the next one at each index of the list we will have 60 elements, we start at the 61st day
X_test= np.array(X_test)


#Reshapping number of predictors we can use, or indicators can add more indicators here
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1)) # this will make it into the correct 3D structure

predicted_stock_price = regressor.predict(X_test)  #store results
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#-------------- Visualizing the results -------------------------------------------------------
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price' ) #plotting Jan 2017
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price' ) #plotting Jan 2017
plt.title('Google Stock Prediction Projects')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


