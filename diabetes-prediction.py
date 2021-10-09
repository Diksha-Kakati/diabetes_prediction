# Import all required libraries 
import pandas as pd  
import streamlit as st 
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns

# Dataset
diabates_dataset = pd.read_csv('C:\\Users\\user\\Desktop\\c++\\git repos\\diabetes_prediction\diabetes.csv')

# Headings 
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Stats for Training Data')
st.write(diabates_dataset.describe())

#Visualition of dataset
st.subheader('Visualisation')
st.bar_chart(diabates_dataset)

# x and y data
x = diabates_dataset.drop(['Outcome'],axis=1)
y = diabates_dataset.iloc[:,-1]
x_train , x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#Function
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0,17,3)
    glucose = st.sidebar.slider('Glucose',0,200,120)
    bp = st.sidebar.slider('BloodPressure',0,122,70)
    skinthickness = st.sidebar.slider('SkinThickness', 0,100,20)
    insulin = st.sidebar.slider('Insulin',0,846,79)
    bmi = st.sidebar.slider('BMI',0,67,20)
    dpf = st.sidebar.slider('DiabetesPedigreeFunction',0.0,2.4,0.47)
    age = st.sidebar.slider('Age',21,88,33)

    user_report = {
        'Pregnancies':pregnancies, 
        'Glucose': glucose,
        'BloodPressure':bp,
        'SkinThickness' : skinthickness,
        'Insulin': insulin,
        'BMI':bmi,
        'DiabetesPedigreeFunction':dpf,
        'Age':age
    }
    report_data = pd.DataFrame(user_report, index=[0])
    return report_data 

# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# Model
rf = RandomForestClassifier()
rf.fit(x_train , y_train)
user_result = rf.predict(user_data)

# OUTPUT 
st.subheader('Accuracy : ')
st.write(str(accuracy_score(y_test, rf.predict(x_test))*100)+'%')

st.subheader("Your Report : ")
output =''

if user_result[0]==0:
    output = 'You are healthy'
else:
    output= 'you are Diabetic'

st.write(output)




