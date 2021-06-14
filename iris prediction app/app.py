import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

st.write("""
# Iris Flower Prediction App

This app predicts the Iris flower type

""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length', 4.3,7.9,5.4)
    sepal_width = st.sidebar.slider('Sepal Width', 2.0,4.4,3.4)
    petal_length = st.sidebar.slider('Petal Length', 1.0,6.9,1.3)
    petal_width = st.sidebar.slider('Petal Width', 0.1,2.5,0.2)
    data = {
        'sepal-length': sepal_length,
        'sepal-width': sepal_width,
        'petal-length': petal_length,
        'petal-width': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = pd.read_csv('iris.csv')
X = iris.drop('species', axis=1)
Y = iris.species

knn = KNeighborsClassifier(5)
svm = SVC()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()

ens = VotingClassifier(estimators = [('KNN', knn),('SVM', svm), ('DT', dt), ('RF', rf)], voting = 'hard')

ens.fit(X, Y)

prediction = ens.predict(df)

st.subheader('Class labels and their corresponding index number')
st.write(iris.species)

st.subheader('Prediction')
st.write(prediction)
