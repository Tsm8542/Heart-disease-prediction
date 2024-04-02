import streamlit as st
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# """Numpy and Pandas for data analysis and data cleaning"""
import numpy as np
import pandas as pd

# """Using matplotlib and seaborn for data visualization"""
import matplotlib.pyplot as plt
import seaborn as sns

# """Using plot_decision_regions to visualize plotted data regions based on decision"""
from mlxtend.plotting import plot_decision_regions


# """Using missingno for visualising of missing values"""
import missingno as msno
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# """K-Nearest Neighbor is a simple classification algorithm"""
from sklearn.neighbors import KNeighborsClassifier

# """Confusion_matrix to find the performance of a model"""
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# """To compare classification models based on classification reports"""
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# """Using GridSearchCV to find optimal hyper parameters for finding most accurate results"""
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

hd_df = pd.read_csv(r'tf.csv')
hd_df_copy = hd_df.copy(deep=True)
hd_df.head()
# print(hd_df.head())

st.title('Heart Disease Checker')

# st.subheader('Training Data')
# st.write(hd_df.describe())

# st.subheader('Visualisation')
# st.bar_chart(hd_df)

sc_X = StandardScaler()
X = pd.DataFrame(sc_X.fit_transform(hd_df_copy.drop(["result"],axis=1),),columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal'])

y = hd_df_copy.result

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)

test_scores = []
train_scores = []

for i in range(1,30):

    knn = KNeighborsClassifier(i)
    knn.fit(X_train,y_train)
    
    train_scores.append(knn.score(X_train,y_train))
    test_scores.append(knn.score(X_test,y_test))

# max_train_score = max(train_scores)
# train_scores_ind = [i for i, v in enumerate(train_scores) if v==max_train_score]
# print('Max train score {} % and k = {}'.format(max_train_score*100,list(map(lambda x: x+1, train_scores_ind))))

# max_test_score = max(test_scores)
# test_scores_ind = [i for i, v in enumerate(test_scores) if v==max_test_score]
# print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))

# knn = KNeighborsClassifier(16)
# knn.fit(X_train,y_train)
# knn.score(X_test,y_test)

# value = 20000
# width = 20000
# plot_decision_regions(X.values,y.values, clf=knn, legend=2,filler_feature_values={2:value,3:value,4:value,5:value,6:value,7:value,8:value,9:value,10:value,11:value,12:value},filler_feature_ranges={2:width,3:width,4:width,5:width,6:width,7:width,8:width,9:width,10:width,11:width,12:width},X_highlight=X_test.values)
# plt.title('KNN with Heart Disease Data')
# plt.show()

# y_pred = knn.predict(X_test)

# cnf_matrix = metrics.confusion_matrix(y_test,y_pred)
# p=sns.heatmap(pd.DataFrame(cnf_matrix),annot=True,cmap="YlGnBu",fmt='g')
# plt.title('Confusion matrix',y=1.1)
# plt.ylabel('Actual lable')
# plt.xlabel('Predicted lable')

# y_pred_proba = knn.predict_proba(X_test)[:,1]
# fpr,tpr,thresholds = roc_curve(y_test, y_pred_proba)

# plt.plot([0,1],[0,1],'k--')
# plt.plot(fpr,tpr,label='Knn')
# plt.xlabel('fpr')
# plt.ylabel('tpr')
# plt.title('Knn(n_neighbors=16) ROC curve')
# plt.show()

# roc_auc_score(y_test,y_pred_proba)

# param_grid = {'n_neighbors':np.arange(1,100)}
# knn = KNeighborsClassifier()
# knn_cv = GridSearchCV(knn,param_grid,cv=5)
# knn_cv.fit(X,y)
# print("Best Score: "+str(knn_cv.best_score_))
# print("Best Parameters: " + str(knn_cv.best_params_))

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


knn = KNeighborsClassifier(n_neighbors=16)
knn.fit(X_train,y_train)
knn.score(X_test,y_test)

import pickle
# pickle.dump(knn,open('HDmodel.pkl','wb'))
loaded_model = pickle.load(open('HDmodel.pkl','rb'))

prediction = loaded_model.predict(user_data)


st.subheader('Accuracy: ')
st.write(str(accuracy_score(y_test,knn.predict(X_test))*100)+'%')

user_result = knn.predict(user_data)
st.subheader('Your Report: ')
output = ''
if prediction[0]==0:
    output = 'You are healthy'
else:
    output = 'You are sick'

st.write(output)