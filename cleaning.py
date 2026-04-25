import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
import pickle

from sklearn.utils.validation import validate_data
from tensorflow.python.profiler.profiler_client import monitor

##loading the data
data=pd.read_csv("Churn_Modelling.csv")


## drop irrelevant features
data=data.drop(["RowNumber", "CustomerId", "Surname"],axis=1)

## handling categorical features by encoding them

## 1.label encoder
label_encoder=LabelEncoder()
data["Gender"]=label_encoder.fit_transform(data["Gender"])


## 2.one hot encoder
from sklearn.preprocessing import OneHotEncoder as OHE
ohe_encoder=OHE(sparse_output=False)
encoded = ohe_encoder.fit_transform(data[["Geography"]])   # double brackets -> 2D one hot encoder expect 2D
# Convert encoded from array to a dataframe without proper column names
encoded_df = pd.DataFrame(encoded, columns=ohe_encoder.get_feature_names_out(["Geography"]))
# remove older feature
data = data.drop("Geography", axis=1)
# concatenate the new dataframe and existing data
data = pd.concat([data, encoded_df], axis=1)


##Save the encoders
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder,file)

with open('ohe_encoder.pkl', 'wb') as file:
    pickle.dump(ohe_encoder,file)

## divide the dataset into independent and dependent features
X=data.drop("Exited",axis=1)
y=data["Exited"]

##Spliting data into test and train data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.35,random_state=55)

##scaling the data
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

## Saving scaler
with open("scaler.pkl",'wb') as file:
    pickle.dump(scaler,file)

##ANN Implementation

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime

##model
model=Sequential(
    [Dense(65,activation='relu',input_shape=(X_train.shape[1],)),     ##hidden layer 1
     Dense(35,activation='relu'),   ##hidden layer 2
     Dense(1,activation='sigmoid')    ## output layer hence only 1 neuron
     ]
)

## total number of parameters
t_param=model.summary()

##compiling model
opt=tf.keras.optimizers.Adam(learning_rate=0.02)     ##in this way can give khudka ka learning_rate
model.compile(optimizer=opt,loss="binary_crossentropy",metrics=['accuracy']) ## here adam can also be directly but fixed learning rate
## setting up tensorboard
log_dir="logs/fit/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorflow_callback=TensorBoard(log_dir=log_dir,histogram_freq=1)
early_stopping_callback=EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True)

## training model
history=model.fit(
    X_train,y_train,validation_data=(X_test,y_test),epochs=100,
    callbacks=[tensorflow_callback,early_stopping_callback]
)
## saving model file
model.save('model.h5')

