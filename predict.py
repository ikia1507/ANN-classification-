import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import numpy as np

##loading model
model=load_model('model.h5')

##loading scaler,encoders
with open('label_encoder.pkl','rb') as file:
    label_encoder=pickle.load(file)

with open('ohe_encoder.pkl','rb') as file:
    ohe_encoder=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


## for new input data
input_data = {
    'CreditScore': 600,
    'Geography': 'France',
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}
input_df=pd.DataFrame([input_data])

geo_encode=ohe_encoder.transform(input_df[['Geography']])
geo_encode_df=pd.DataFrame(geo_encode,columns=ohe_encoder.get_feature_names_out(['Geography']))

input_df = input_df.drop("Geography", axis=1)
input_df=pd.concat([input_df.reset_index(drop=True),geo_encode_df],axis=1)

input_df["Gender"]=label_encoder.transform(input_df['Gender'])


##scaling part
input_scaled=scaler.transform(input_df)

##prediction
prediction=model.predict(input_scaled)

if prediction[0][0]>0.5:
    print("will not leave")
else:
    print("will leave")
