import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train a RandomForest Classifier
clf = RandomForestClassifier()
clf.fit(X, y)

# Define a function to take user input for the Iris flower features
def user_input_features():
    st.sidebar.header('Input Features')
    
    sepal_length = st.sidebar.slider(
        'Sepal length (cm)',
        min_value=float(X[:,0].min()),
        max_value=float(X[:,0].max()),
        value=float(X[:,0].mean()),
        step=0.1
    )
    sepal_width = st.sidebar.slider(
        'Sepal width (cm)',
        min_value=float(X[:,1].min()),
        max_value=float(X[:,1].max()),
        value=float(X[:,1].mean()),
        step=0.1
    )
    petal_length = st.sidebar.slider(
        'Petal length (cm)',
        min_value=float(X[:,2].min()),
        max_value=float(X[:,2].max()),
        value=float(X[:,2].mean()),
        step=0.1
    )
    petal_width = st.sidebar.slider(
        'Petal width (cm)',
        min_value=float(X[:,3].min()),
        max_value=float(X[:,3].max()),
        value=float(X[:,3].mean()),
        step=0.1
    )
    
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
df = user_input_features()

# Display the user input
st.subheader('User Input Features')
st.write(df)

# Predict the class of the input features
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# Display the prediction and corresponding probability
st.subheader('Prediction')
st.write(f'The predicted Iris species is: **{iris.target_names[prediction][0]}**')

st.subheader('Prediction Probability')
prob_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
prob_df = prob_df.rename(columns={name: f"{name} Probability" for name in prob_df.columns})
st.write(prob_df)
