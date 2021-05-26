import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

st.set_option('deprecation.showPyplotGlobalUse', False)


st.title("Application Web de Machine Learning pour la Détection de Fraude par Carte de Crédit 💳")
st.subheader("Auteur : Josué Afouda")
st.sidebar.title('Classification Binaire Automatique 💳')
st.markdown("Cette application est destinée à la détection de fraude par des algorithmes de Machine Learning")
st.sidebar.markdown("Est-ce une transaction frauduleuse ou pas ? 💳")

# Fonction de chargement des donnees
@st.cache(persist = True)
def load_data():
    data = pd.read_csv("creditcard.csv")
    return data
    
# Affichage de la table des donnees
df = load_data()
df_sample = df.sample(100)
if st.sidebar.checkbox("Données brutes",False):
    st.subheader("Jeu de données 'Credit Card' : Echantillon de 100 observations")
    st.write(df_sample)

seed = 123
# Creation d'un train set et d'un test set
@st.cache(persist = True)
def split(df):
    y = df.Class
    X = df.drop('Class', axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state = seed)
    return X_train, X_test, y_train, y_test

    # Model Evaluation Metrics
def plot_metrics(metrics_list):
    if 'Confusion Matrix' in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, X_test, y_test, display_labels = class_names)
        st.pyplot()

    if 'ROC Curve' in metrics_list:
        st.subheader("ROC Curve")
        plot_roc_curve(model, X_test, y_test)
        st.pyplot()

    if 'Precision-Recall Curve' in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(model, X_test, y_test)
        st.pyplot()

X_train, X_test, y_train, y_test = split(df)
class_names = ['T. Authenthique', 'T. Frauduleuse']
st.sidebar.subheader("Choisir un Classificateur")
classifier = st.sidebar.selectbox("Classificateur",("Random Forest", "Support Vector Machine (SVM)", "Logistic Regression"))

    
# Random Forest
if classifier == "Random Forest":
    st.sidebar.subheader("Hyperparamètres du modèle")
    n_estimators = st.sidebar.number_input("Nombre d'arbres dans la forêt", 100, 5000, step = 10, key = 'n_etimators')
    max_depth = st.sidebar.number_input("Profondeur maximale d'un arbre", 1, 20, step = 1, key = 'max_depth')
    bootstrap = st.sidebar.radio("Échantillons Bootstrap lors de la création d'arbres", ('True', 'False'), key = 'bootstrap')

    # Train a Logistic Regression Classifier
    metrics = st.sidebar.multiselect("Choisir une métrique d'évaluation", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

    if st.sidebar.button("Execution", key = 'classify'):
        st.subheader("Random Forest Results")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                           bootstrap=bootstrap, random_state=seed, n_jobs=-1)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
        plot_metrics(metrics)                                 
                                      
                                 
    # SVM
if classifier == "Support Vector Machine (SVM)":
    st.sidebar.subheader("Hyperparamètres du modèle")
    C = st.sidebar.number_input("C (Paramètre de Régularisation)", 0.01, 10.0, key = 'C')
    kernel = st.sidebar.radio("kernel", ("rbf", "linear"), key = 'kernel')
    gamma = st.sidebar.radio("Gamma (Kernel Coefficient)",("scale", "auto"), key = 'gamma')

    # Train a SVM Classifier
    metrics = st.sidebar.multiselect("Choisir une métrique d'évaluation", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

    if st.sidebar.button("Execution", key = 'classify'):
        st.subheader("Support Vector Machine (SVM) Results")
        model = SVC(C=C, kernel=kernel, gamma=gamma)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
        plot_metrics(metrics)


    # Logistic Regression
if classifier == "Logistic Regression":
    st.sidebar.subheader("Hyperparamètres du modèle")
    C = st.sidebar.number_input("C (Paramètre de Régularisation)", 0.01, 10.0, key = 'C_LR')
    max_iter = st.sidebar.slider("Nombre maximum d'itérationse", 100, 500, key = 'max_iter')

    # Train a Logistic Regression Classifier
    metrics = st.sidebar.multiselect("Choisir une métrique d'évaluation", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

    if st.sidebar.button("Execution", key = 'classify'):
        st.subheader("Logistic Regression Results")
        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        y_pred = model.predict(X_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
        plot_metrics(metrics)

