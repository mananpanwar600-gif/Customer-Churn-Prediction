import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import pandas as pd
import numpy as np
import pickle

model = tf.keras.models.load_model('artifacts/model.h5')

with open('artifacts/encoder.pkl','rb') as file:
    encoder = pickle.load(file)

with open('artifacts/scaler.pkl','rb') as file:
    scaler = pickle.load(file)

## streamlit app

st.title('Customer Churn Prediction')

## User inputs

geography = st.selectbox('Geography',encoder.categories_[1])
gender = st.selectbox('Gender',encoder.categories_[0])
age = st.slider('Age',18,90)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number Of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

# prepare input data
input_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

encoded = encoder.transform([[gender,geography]])
encoded_df = pd.DataFrame(encoded,columns=encoder.get_feature_names_out(['Gender','Geography']))

input_df = input_df.drop(['Geography', 'Gender'], axis=1)

input_df = pd.concat([input_df.reset_index(drop=True),
                      encoded_df.reset_index(drop=True)],
                     axis=1)

input_data_scaled = scaler.transform(input_df)

## Prediction
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability : {prediction_proba:.2f}')

if prediction_proba>0.5:
    st.write('The Customer is likely to churn')
else:
    st.write('The Customer is not likely to churn')

