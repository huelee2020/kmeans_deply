import streamlit as st
from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#load model from joblib file
km_model = load('./kmeans_model.ml')

#create an web app
st.title("Clustering App using K-Means Algorithm")

# create streamlit widgets
# insert data by uploading a csv file
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, index_col=0)

    clustering_cols = ['bmxleg', 'bmxwaist']
    data = data.dropna(subset=clustering_cols)
    X = data[['bmxleg', 'bmxwaist']].dropna()
    
    if data.shape[0] > km_model.n_clusters:
        cluster_lb = km_model.predict(data[clustering_cols])

        # plot the clusters
        data['cluster'] = cluster_lb
        st.write(data)

        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.write("Plotting graph")
        plt.figure(figsize=(6, 6))
        sns.scatterplot(data=data, x='bmxleg', y='bmxwaist', hue='cluster')
        plt.title("K-Means Clustering")
        st.pyplot()

else:
    st.write("Please upload a CSV file to predict the clusters")