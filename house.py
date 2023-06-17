'''House Rent Prediction The rent of a house depends on a lot of factors. With appropriate data and Machine Learning techniques, many real estate platforms find the housing options according to the customer’s budget. So, if you want to learn how to use Machine Learning to predict the rent of a house, this article is for you. In this article, I will take you through the task of House Rent Prediction with Machine Learning using Python.

The rent of a housing property depends on a lot of factors like:

1)number of bedrooms, hall, and kitchen

2)size of the property

3)the floor of the house

4)area type

5)area locality

6)City

7)furnishing status of the house To build a house rent prediction system, we need data based on the factors affecting the rent of a housing property. I found a dataset from Kaggle which includes all the features we need.

Dataset :- https://www.kaggle.com/datasets/iamsouravbanerjee/house-rent-prediction-dataset '''     
    
    #####    source code    #####
    
#     step1:
#importing libraries
import numpy as np #2 perform mathematical operations on arrays
import pandas as pd #for data analysis

#statistical graphics
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split #measure the accuracy of the model
from keras.models import Sequential #allows u 2 create models layers by layers, hvg a stack of layers(as it is stack it'll acc every layer as 1 element)
from keras.layers import Dense #all the layers frm 1 layer r connected to 2nd layer
from keras.layers import LSTM # Long Short Term Memory RNN model, used in Time Series


#         step2:-
#load the dataset 2 pandas data frame for manupulating the data
raw_data = pd.read_csv('House_Rent_Dataset.csv', encoding = 'latin-1')

#now v hv 2 replace null values with null string otherwise it will show errors
#v will store this in variable claaed "mail_data"
data = raw_data.where((pd.notnull(raw_data)), '')

#lets check the shape of the dataset
data.shape


#      step3:
# printing the first 10 rows of the dataset
data.head(10)



#           step4:
# printing the last 10 rows of the dataset
data.tail(10)



#          step5:
#dataset informations
data.info()


#       step6:
#data preprocessing 2 check whether if there r any empty values
#checking the number of missing values in each column
data.isnull().sum()


#      step7:
#statistical measures about the rent
print(f"Mean Rent: {data.Rent.mean()}")
print(f"Median Rent: {data.Rent.median()}")
print(f"Highest Rent: {data.Rent.max()}")
print(f"Lowest Rent: {data.Rent.min()}")




######      Visualization        #####
#   step1:
#printing the rent in different cities acc to BHK
figure = px.bar(data, x=data["City"],
                y = data["Rent"],
                color = data["BHK"],
            title="Rent in Different Cities According to BHK")
figure.show()




#         step2:
#printing the rent in different cities acc to Area type
figure = px.bar(data, x=data["City"],
                y = data["Rent"],
                color = data["Area Type"],
            title="Rent in Different Cities According to Area Type")
figure.show()



#           step3:
#printing the rent in different cities acc to Furniture status
figure = px.bar(data, x=data["City"],
                y = data["Rent"],
                color = data["Furnishing Status"],
            title="Rent in Different Cities According to Furnishing Status")
figure.show()



#            step 4:
#printing the rent in different cities acc to Size
figure = px.bar(data, x=data["City"],
                y = data["Rent"],
                color = data["Size"],
            title="Rent in Different Cities According to Size")
figure.show()



#    step5:
#printing the no of houses available for rent
cities = data["City"].value_counts()
label = cities.index
counts = cities.values
colors = ['gold','lightgreen']

fig = go.Figure(data=[go.Pie(labels=label, values=counts, hole=0.5)])
fig.update_layout(title_text='Number of Houses Available for Rent')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30,
                  marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()







#  step6:
#printing the Preference of Tenant
tenant = data["Tenant Preferred"].value_counts()
label = tenant.index
counts = tenant.values
colors = ['gold','lightgreen']

fig = go.Figure(data=[go.Pie(labels=label, values=counts, hole=0.5)])
fig.update_layout(title_text='Preference of Tenant in India')
fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=30,
                  marker=dict(colors=colors, line=dict(color='black', width=3)))
fig.show()



####          Preprocessing             ###
#  step1:
#Now let’s prepare the data for the task of training a deep learning model.
#Here I will convert all the categorical features into numerical values:
data["Area Type"] = data["Area Type"].map({"Super Area": 1,
                                           "Built Area": 2,
                                           "Carpet Area": 3})

data["City"] = data["City"].map({"Mumbai": 400000, "Chennai": 600000,
                                 "Bangalore": 560000, "Hyderabad": 500000,
                                 "Delhi": 110000, "Kolkata": 700000})

data["Furnishing Status"] = data["Furnishing Status"].map({"Unfurnished": 0,
                                                           "Semi-Furnished": 1,
                                                           "Furnished": 2})

data["Tenant Preferred"] = data["Tenant Preferred"].map({"Bachelors": 1,
                                                         "Bachelors/Family": 2,
                                                         "Family": 3})

#     step 2:
# printing the first 10 rows of the dataset
data.head(10)



#       step3:
# printing the last 10 rows of the dataset
data.tail(10)



#       step4:
#statistical measures about the data
data.describe(include = 'all')


###       Splitting the data          ###

###   Splitting the data into Features & Targets   ###

# step1:
#assigning features as X
x = np.array(data[["BHK", "Size", "Area Type", "City", "Furnishing Status", "Tenant Preferred", "Bathroom"]])

#assigning targets as Y
y = np.array(data[["Rent"]])


#     step2:
print(x) #printing the features
print("---------------------------------------------------------------------------------------------------------------------------")
print(y) #printing the targets



###       Splitting the data into Training and Testing        ####


#    spliting the dataset in2 Training & Testing


#test size --> 2 specify the percentage of test data needed ==> 0.2 ==> 20%
#    step1:
#random state --> specific split of data each value of random_state splits the data differently, v can put any state v want
#v need 2 specify the same random_state everytym if v want 2 split the data the same way everytym
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 2)

#      step2:
#checking dimensions of Features
print(x.shape, x_train.shape, x_test.shape)

# step3:
#checking dimensions of Targets
print(y.shape, y_train.shape, y_test.shape)


########         LSTM Deep Learning Model           #####
#   step1:
#now let’s train a house rent prediction model using an LSTM neural network model
#These are the building blocks of neural networks that will learn how to predict x_train based on its input shape (x_train's shape)
#train a model with an LSTM hidden layer and Dense output layer
model = Sequential()

#v add 2 LSTMs with 128 and 64 units respectively
#The first dimension of the input is x_train, which has a shape of (n_samples, n_features) where n_samples is the number of samples in the training set and n_features is the number of features in each sample
#The second dimension of the input is 1, which means that there are no dimensions
model.add(LSTM(128, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))

#we add two Dense layers with 25 and 1 units respectively which will help us compute our predictions using backpropagation through time (BPTT) later on when we train our network
#Dense Layer is simple layer of neurons in which each neorons receives input frm all the neurons of previous layer
model.add(Dense(25))
model.add(Dense(1))

#we'll c the summary of our model
model.summary()


#       step2:
#for better accuracy v need 2 scale the dataset
#v choose adam optimizer as it is the best
model.compile(optimizer='adam', loss='mse')
#v r fitting our train data 2 the model
#fit is where the training actually hppns
#epochs is no of iterations for which ur neural network is gng 2 run the train
#v took epochs as 36 coz it is suggested to take 3 tyms our data column count
#as our data column count is 12 => 12 * 3 = 36 :)
model.fit(x_train, y_train, batch_size=52, epochs=36)



#     step3:
#lets c the loss on our test dataset
model.evaluate(x_test, y_test)




######          Predictive Model        #####
#    Prediction Model
#      step1:
print("Enter House Details to Predict Rent")
print('----------------------------------------')

a = int(input("Number of BHK: "))
b = int(input("Size of the House in Sqrt: "))
c = int(input("Area Type (Super Area = 1, Built Area = 2, Carpet Area = 3): "))
d = int(input("Pin Code of the City: "))
e = int(input("Furnishing Status of the House (Unfurnished = 0, Semi-Furnished = 1, Furnished = 2): "))
f = int(input("Tenant Type (Bachelors = 1, Bachelors/Family = 2, Only Family = 3): "))
g = int(input("Number of bathrooms: "))

features = np.array([[a, b, c, d, e, f, g]])

print("Predicted House Price = ", model.predict(features))2



'''   Summary :
 So this is how to use Machine Learning to predict the rent of a housing property. With appropriate data and Machine Learning techniques, many real estate platforms find the housing options according to the customer’s budget.'''




































