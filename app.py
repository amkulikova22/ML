import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pickle
from sklearn.linear_model import LinearRegression

st.title('Hello World')
file = st.file_uploader("Upload a file", type='csv')

if file is not None:
    df = pd.read_csv(file)
    st.dataframe(df)

if st.checkbox('Show dataframe'):
    # st.header("EDA")
    fig = sns.pairplot(df)
    st.pyplot(fig)
    fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include='number').corr(), ax=ax)
    # ax.set_title("Heatmap с Seaborn")
    st.pyplot(fig2)
    df = df.drop(columns=['Unnamed: 0', 'name', 'fuel', 'seller_type', 'transmission', 'owner'])
    model = pickle.load(open('/Users/anastasia/Documents/model.pkl', 'rb'))
    y_pred_linear = model.predict(df)
    y_pred_linear
    # fig3 = plt.bar(x=model.feature_names_in_, height=model.coef_)
    # st.pyplot(fig3)
    fig3, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x=model.feature_names_in_, height=model.coef_)
    st.pyplot(fig3)
