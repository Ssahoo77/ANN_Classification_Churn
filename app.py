import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

model=tf.keras.models.load_model('model.h5')
with open("label_encoder_gender.pkl", 'rb') as file:
    label_encoder_gender=pickle.load(file)
    
with open("oneHotEncoder_geo.pkl", 'rb') as file:
    label_encoder_geo=pickle.load(file)

with open("scalar.pkl", 'rb') as file:
    scalar=pickle.load(file)


st.title('Customer Churn Prediction')

geography= st.selectbox('Geography', label_encoder_geo.categories_[0])
gender= st.selectbox('Gender', label_encoder_gender.classes_)
age=st.slider('Age', 18, 92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure', 0,10)
num_of_prod=st.slider('Number Of Products', 1,4)
hasCreditCard=st.selectbox('Has Credit Card', [0,1])
is_Active_member= st.selectbox('Is Active Member',[0,1])


input_data= pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_prod],
    'HasCrCard': [hasCreditCard],
    'IsActiveMember': [is_Active_member],
    'EstimatedSalary': [estimated_salary]
})
geo_encoded= label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

input_df=pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_scaled= scalar.transform(input_df)

prediction = model.predict(input_scaled)
prediction_prob=prediction[0][0]

st.write(f'Churn Probability : {prediction_prob}')

if prediction_prob>0.5:
    st.write('Customer is likely to churn.')
else:
    st.write('Customer is not likely to churn.')
