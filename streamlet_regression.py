import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle

model = tf.keras.models.load_model('regression_model.h5')

with open('geo_encoder.pkl', 'rb') as f:
    ohe_geo = pickle.load(f)           # OneHotEncoder from training
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder_gender.pkl', 'rb') as f:
    le_gender = pickle.load(f)         # LabelEncoder from training

st.title('Customer Salary Prediction')

# --- Inputs (match training features) ---
geography = st.selectbox('Geography', ohe_geo.categories_[0])
gender = st.selectbox('Gender', le_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', value=0.0)
credit_score = st.number_input('Credit Score', value=650)
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
exited = st.selectbox('Exited', [0, 1])

# --- Build row in the SAME order as training ---
base = pd.DataFrame([{
    'CreditScore': credit_score,
    'Gender': le_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'Exited': exited
}])

# One-hot Geography (same encoder)
geo_encoded = ohe_geo.transform([[geography]]).toarray()
geo_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))

# Combine
row = pd.concat([base.reset_index(drop=True), geo_df], axis=1)

# Scale with TRAINED scaler (no fitting)
row_scaled = scaler.transform(row)

# Predict
pred = model.predict(row_scaled)
st.write(f'Predicted Salary: {float(pred[0][0]):.2f}')
