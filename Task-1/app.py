import streamlit as st
#from sklearn.datasets import load_wine, load_breast_cancer, load_iris, load_digits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.datasets import make_moons, make_circles
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn import metrics


st.title("Classification WebApp")

st.sidebar.subheader("Choose Dataset:")
dataset_name = st.sidebar.selectbox("Dataset:", ("make_moons","make_circles"))

st.sidebar.subheader("Set Test Size:")
split_ratio = st.sidebar.slider("% Test Size", 0.1, 0.9, step=0.1)

st.sidebar.subheader("Choose Classifier:")
classifier_name = st.sidebar.selectbox("Classifier:", ("Gradient Boost","Random Forest"))


def tune_parameter(classifier_name):
    parameters = dict()

    if classifier_name == "Gradient Boost":

        st.sidebar.subheader("Model Hyperparameters:")

        lr = float(st.sidebar.number_input("learning_rate", 0.1, 1.0))
        parameters["learning_rate"] = lr

        estimators = int(st.sidebar.number_input("n_estimators", 1, 2000))
        parameters["n_estimators"] = estimators

        depth = st.sidebar.slider("max_depth", 2, 30)
        parameters["max_depth"] = depth

        criterion = st.sidebar.radio("Select criterion", ("friedman_mse","mse"))
        parameters["Select criterion"] = criterion

        samples = st.sidebar.slider("sub_samples", 0.1, 1.0)
        parameters["sub_samples"] = samples

        split = int(st.sidebar.number_input("min_samples_split", 2, 100))
        parameters["min_samples_split"] = split

        leaf = int(st.sidebar.number_input("min_samples_leaf", 1, 100))
        parameters["min_samples_leaf"] = leaf

        weight_fraction = st.sidebar.slider("min_weight_fraction_leaf", 0.0, 1.0)
        parameters["min_weight_fraction_leaf"] = weight_fraction

        decrease = st.sidebar.slider("min_impurity_decrease", 0.0, 1.0)
        parameters["min_impurity_decrease"] = decrease

        features = st.sidebar.radio("Select max features", ("auto","sqrt","log2"))
        parameters["Select max features"] = features

        warm = st.sidebar.radio("warm_start", ("False","True"))
        parameters["warm_start"] = warm

        fraction = st.sidebar.slider("validation_fraction", 0.1, 1.0)
        parameters["validation_fraction"] = fraction

        alpha = st.sidebar.slider("ccp_alpha", 0.0, 1.0)
        parameters["ccp_alpha"] = alpha

        state = st.sidebar.slider("random_state", 0, 100)
        parameters["random_state"] = state

    else:

        st.sidebar.subheader("Model Hyperparameters:")

        estimators = int(st.sidebar.number_input("n_estimators", 1, 2000))
        parameters["n_estimators"] = estimators

        depth = st.sidebar.slider("max_depth", 2, 30)
        parameters["max_depth"] = depth

        criterion = st.sidebar.radio("Select criterion", ("gini","entropy"))
        parameters["Select criterion"] = criterion

        split = int(st.sidebar.number_input("min_samples_split", 2, 100))
        parameters["min_samples_split"] = split

        leaf = int(st.sidebar.number_input("min_samples_leaf", 1, 100))
        parameters["min_samples_leaf"] = leaf

        weight_fraction = st.sidebar.slider("min_weight_fraction_leaf", 0.0, 1.0)
        parameters["min_weight_fraction_leaf"] = weight_fraction

        decrease = st.sidebar.slider("min_impurity_decrease", 0.0, 1.0)
        parameters["min_impurity_decrease"] = decrease

        bootstrap = st.sidebar.radio("bootstrap", ("True","False"))
        parameters["bootstrap"] = bootstrap

        features = st.sidebar.radio("Select max features", ("auto","sqrt","log2"))
        parameters["Select max features"] = features

        warm = st.sidebar.radio("warm_start", ("False","True"))
        parameters["warm_start"] = warm

        alpha = st.sidebar.slider("ccp_alpha", 0.0, 1.0)
        parameters["ccp_alpha"] = alpha

        state = st.sidebar.slider("random_state", 0, 100)
        parameters["random_state"] = state

    return parameters

parameters = tune_parameter(classifier_name)


def return_data(dataset):
    
    if dataset == 'make_moons':
        st.subheader("make_moons")
        X, y = make_moons(n_samples=1000, noise=0.30, random_state=42)
        df = pd.DataFrame(X)
        df['y'] = y

        
    else:
        st.subheader("make_circles")
        X, y = make_circles(n_samples=1000, noise=0.30, random_state=42)
        df = pd.DataFrame(X)
        df['y'] = y
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=split_ratio)


    return X_train, X_test, y_train, y_test, df, X, y

X_train, X_test, y_train, y_test, df, X, y = return_data(dataset_name)


if(st.checkbox("Show the dataset", False)):
    
    st.dataframe(df.style.hide_index())
    st.write("Number of rows:", df.shape[0])
    st.write("Number of columns:", df.shape[1])

fig, ax = plt.subplots()

# Plot initial graph
ax.scatter(X.T[0], X.T[1], c=y, cmap='viridis')
orig = st.pyplot(fig)


if(st.sidebar.button("Run Algorithm")):
    if classifier_name == "Gradient Boost":
        clf = GradientBoostingClassifier(learning_rate=parameters["learning_rate"], n_estimators=parameters["n_estimators"], max_depth=parameters["max_depth"], 
                                        criterion=parameters["Select criterion"], subsample=parameters["sub_samples"], min_samples_split=parameters["min_samples_split"], min_samples_leaf=parameters["min_samples_leaf"],
                                        min_weight_fraction_leaf=parameters["min_weight_fraction_leaf"], min_impurity_decrease=parameters["min_impurity_decrease"], warm_start=parameters["warm_start"], max_features=parameters["Select max features"], 
                                        validation_fraction=parameters["validation_fraction"], ccp_alpha=parameters["ccp_alpha"], random_state=parameters["random_state"])

    else:
        clf = RandomForestClassifier(n_estimators=parameters["n_estimators"], max_depth=parameters["max_depth"], 
                                    criterion=parameters["Select criterion"], min_samples_split=parameters["min_samples_split"], min_samples_leaf=parameters["min_samples_leaf"],
                                    min_weight_fraction_leaf=parameters["min_weight_fraction_leaf"], min_impurity_decrease=parameters["min_impurity_decrease"], warm_start=parameters["warm_start"],
                                    bootstrap=parameters["bootstrap"], max_features=parameters["Select max features"], ccp_alpha=parameters["ccp_alpha"], random_state=parameters["random_state"])

    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    orig.empty()
    a = accuracy_score(y_test, y_pred)
    confMatrix = confusion_matrix(y_test, y_pred) 
    kappa_score = cohen_kappa_score(y_test, y_pred)     

    st.write("Classifier name:", classifier_name)
    st.write("Accuracy:", a)
    st.write('Kappa Score : ', kappa_score)
    st.write('Confusion Matrix :\n', confMatrix)
    
    a = np.linspace(X.min(), X.max(), 100)
    
    XX, YY = np.meshgrid(a, a)
    labels = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    labels = labels.reshape(XX.shape)
    ax.contourf(XX, YY, labels, alpha=0.2)
    ax.scatter(X[:,0], X[:,1], c=y, cmap='viridis', alpha=0.7)
    plt.xlabel("Col1")
    plt.ylabel("Col2")
    orig = st.pyplot(fig)

    
    