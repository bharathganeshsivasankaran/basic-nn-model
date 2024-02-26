# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The problem statement for developing a neural network regression model involves predicting a continuous value output based on a set of input features. In regression tasks, the goal is to learn a mapping from input variables to a continuous target variable.

## Neural Network Model

![1](https://github.com/Ashwinkumar-03/basic-nn-model/assets/118663725/da0174dd-f75f-4dfe-84f7-3c30ffcadef8)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Ashwin Kumar. S
### Register Number: 212222240013
```
#DEPENDENCIES:

from google.colab import auth
import gspread
from google.auth import default

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential as Seq
from tensorflow.keras.layers import Dense as Den
from tensorflow.keras.metrics import RootMeanSquaredError as rmse

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

#DATA FROM SHEETS:

worksheet = gc.open("DL ex 1").sheet1
rows=worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'Input':'int'})
df = df.astype({'Output':'int'})
print(df)

df.head()

#DATA VISUALIZATION:

 x = df[["Input"]] .values
 y = df[["Output"]].values

#DATA SPLIT AND PREPROCESSING:

scaler = MinMaxScaler()
scaler.fit(x)
x_n = scaler.fit_transform(x)
x_train,x_test,y_train,y_test = train_test_split(x_n,y,test_size = 0.3,random_state = 3)

print(x_train)
print(x_test)

#REGRESSIVE MODEL:

 model = Seq([
 Den(4,activation = 'relu',input_shape=[1]),
 Den(6),
 Den(3,activation = 'relu'),
 Den(1),
 ])

 model.compile(optimizer = 'rmsprop',loss = 'mse')
 model.fit(x_train,y_train,epochs=20)
 model.fit(x_train,y_train,epochs=20)

#LOSS CALCULATION:

loss_plot = pd.DataFrame(model.history.history)
loss_plot.plot()

 err = rmse()
 preds = model.predict(x_test)
 err(y_test,preds)

 x_n1 = [[30]]
 x_n_n = scaler.transform(x_n1)
 model.predict(x_n_n)

#PREDICTION:

y_pred=model.predict(x_test)
y_pred

```
## Dataset Information

![image](https://github.com/Ashwinkumar-03/basic-nn-model/assets/118663725/d2dbc163-4a2f-4412-9781-7b0d3cc39a82)

## OUTPUT

### Head():
![image](https://github.com/Ashwinkumar-03/basic-nn-model/assets/118663725/5b6a18d6-a4f3-4dc6-8bba-9977c3b66f34)

### value of X_train and X_test:

![image](https://github.com/Ashwinkumar-03/basic-nn-model/assets/118663725/34a2c67f-f552-4b4a-acde-3916ecbe0f87)

### ARCHITECTURE AND TRAINING:

![image](https://github.com/Ashwinkumar-03/basic-nn-model/assets/118663725/1f574f85-9e73-414d-bcab-752ea3248266)

![image](https://github.com/Ashwinkumar-03/basic-nn-model/assets/118663725/5f912d3f-9de6-497b-86c6-af32c905ad38)


### Training Loss Vs Iteration Plot

![image](https://github.com/Ashwinkumar-03/basic-nn-model/assets/118663725/917afe3b-d8a6-48e5-b80f-4875973f73e2)

### Test Data Root Mean Squared Error

![image](https://github.com/Ashwinkumar-03/basic-nn-model/assets/118663725/9dacefe8-186a-4cde-834b-635ab6fa74b8)

![image](https://github.com/Ashwinkumar-03/basic-nn-model/assets/118663725/0fe0d0ba-3b9e-43f6-9884-cfa32d1f4026)

### New Sample Data Prediction

![image](https://github.com/Ashwinkumar-03/basic-nn-model/assets/118663725/3c1d9826-e4ac-4925-b66c-82bed963661a)

## RESULT

A neural network regression model for the given dataset is developed .

