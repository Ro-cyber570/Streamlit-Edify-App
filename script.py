from turtle import color
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as qo
from sklearn import datasets, neighbors
from mlxtend.plotting import plot_decision_regions
import json
from urllib.request import urlopen
import seaborn
from sklearn.decomposition import PCA

st.title("KNN")
uploaded_file = st.file_uploader(label="Upload CSV file",type=['csv','xlsx'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.header("Dataset")
    a1=st.number_input('Pick a number of rows', 0, 10000)
    if a1>=1:
        st.dataframe(data.head(a1))

        
        list1=[]  
        for i in data.columns:
            list1.append(i)
        target=st.selectbox("What is the target column?",(list1))
        x=data.loc[:, data.columns != target]
        y=data[target]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)
        nn=st.number_input('Pick a KNN number', 1, 1000)
        weights=st.selectbox("What is the weight type",("uniform","distance"))
        knn = KNeighborsClassifier(n_neighbors=nn,weights=weights)
        knn.fit(X_train,y_train)
        y_pred=knn.predict(X_test)

        check1 = st.checkbox("Accuracy Score")
        if check1:
            st.header("Accuracy Score")
            st.subheader(metrics.accuracy_score(y_test,y_pred))

        check2 = st.checkbox("HeatMap")
        if check2:
            fig, ax = plt.subplots()
            sns.heatmap(data.corr(), ax=ax)
            st.header("Heatmap")
            st.write(fig)
        
        check3 = st.checkbox("SwarmPlot")
        if check3:
            fig = plt.figure(figsize=(10, 5))
            sns.swarmplot(y_test,y_pred)
            st.header("Swarmplot")
            st.pyplot(fig)

        check4 = st.checkbox("StripPlot")
        if check4:
            fig = plt.figure(figsize=(10, 8))
            seaborn.stripplot(data=data, size=3, edgecolor='blue')
            st.header("Stripplot")
            st.pyplot(fig)
        check5 = st.checkbox("KNN-Visualization")
        if check5:
            x = data[["Glucose","Insulin"]].values
            y = data["Outcome"].astype(int).values
            knn.fit(x,y)
            fig=plt.figure(figsize=(10, 5))
            plot_decision_regions(x, y, clf=knn, legend=2)
            plt.xlabel("Insulin")
            plt.ylabel("Glucose")
            st.pyplot(fig)

        # check = st.checkbox('Accuracy Score')
        # if check:
        #     st.write('hello box')