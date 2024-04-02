import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



df = pd.read_csv(r'tf.csv')
st.title('Heart Disease Checkup')

st.subheader('Training Data')
st.write(df.describe())

st.subheader('Visualisation')
st.bar_chart(df)

x = df.drop(['result'],axis=1)
y = df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

def user_report():
    age = st.text_input('Age:',25)
    sex = st.radio('Sex:',(0,1),captions=["Male","Female"])
    cp = st.radio('Chest pain:',(1,2,3,4),captions= ["Typical Angina","Atypical Angina","Non-anginal pain","Asymptomatic"])
    trestbps = st.text_input('Trestbps:',94,key="Resting blood pressure")
    chol = st.text_input('Chol:',200,key="Serum cholestoral in mg/dl")
    fbs = st.radio('Fbs:',(0,1),captions=["False","True"])
    restecg = st.radio('Restecg:',(0,1,2),captions=["Normal","Having ST-T wave abnormality","Showing probable or definite left ventricular hypertrophy by Estes"])
    thalach = st.text_input('Thalach:',96,key="Maximum heart rate achieved")
    exang = st.radio('Exang:',(0,1),captions=["No","Yes"])
    oldpeak = st.text_input('oldpeak:',4,key="ST depression induced by exercise relative to rest")
    slope = st.radio('slope:',(1,2,3),captions=["Upsloping","Flat","Downsloping"])
    ca = st.radio('ca:',(0,1,2,3))
    thal = st.radio('thal:',(3,6,7),captions=["Normal","Fixed defect","Reversable defect"])

    user_report_data = {
        'age':age,
        'sex':sex,
        'cp':cp,
        'trestbps':trestbps,
        'chol':chol,
        'fbs':fbs,
        'restecg':restecg,
        'thalach':thalach,
        'exang':exang,
        'oldpeak':oldpeak,
        'slope':slope,
        'ca':ca,
        'thal':thal
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

rf = RandomForestClassifier()
rf.fit(x_train,y_train)

st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test,rf.predict(x_test))*100)+'%')

user_result = rf.predict(user_data)
st.subheader('Your Report: ')
output = ''
if user_result[0]==0:
    output = 'You are healthy'
else:
    output = 'You are sick'

st.write(output)