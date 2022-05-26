import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("advertising.csv")
#use required features

# Setting the value for X and Y
x = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

# creating an object of LinearRegression class
LR = LinearRegression()
#Fitting the Multiple Linear Regression model
LR.fit(x_train,y_train)

# Accuracy of the model is 91.16%

#saving model to current directory
#pickle serializes objects so they can be save to a file, and loaded in a program again later on
pickle.dump(LR, open('model.pkl', 'wb'))

