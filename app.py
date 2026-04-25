##streamlit web interface
from copyreg import pickle

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pickle

from cleaning import ohe_encoder

## LOADING TRAINED MODEL
model=tf.keras.models.load_model('model.h5')

##LOADING ENCODER AND SCALER
with open('ohe_encoder.pkl','rb') as file:
    geo_ohe=pickle.load(file)
with open('label_encoder.pkl','rb') as file:
    gender_label=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

##streamlit app
st.title('Customer Churn Prediction')
# User input
geography = st.selectbox('Geography', geo_ohe.categories_[0])
gender = st.selectbox('Gender', gender_label.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_label.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encode=ohe_encoder.transform([[geography]])
geo_encode_df=pd.DataFrame(geo_encode,columns=ohe_encoder.get_feature_names_out(['Geography']))

input_data=pd.concat([input_data.reset_index(drop=True),geo_encode_df],axis=1)
input_data=scaler.tranform(input_data)
prediction=model.predict(input_data)
predict_proba=prediction[0][0]
if predict_proba>0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")


