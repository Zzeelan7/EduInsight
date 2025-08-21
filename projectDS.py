import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import eli5
from lime.lime_tabular import LimeTabularExplainer

# import kagglehub

# path = kagglehub.dataset_download("jayaantanaath/simulated-dataset-jee-dropout-after-class-12")

df = pd.read_csv("JEE_Dropout_After_Class_12.csv")

df["final_score"] = (0.5*df['jee_main_score']+
                     0.3*df['jee_advanced_score']+
                     0.2*df['mock_test_score_avg'])
bins = [df['final_score'].min(), (df['final_score'].mean()+df['final_score'].min())/2,(df['final_score'].mean()+df['final_score'].max())/2 , df['final_score'].max()]
# labels = ["At Risk", "Average", "High Performer"]
labels = [0, 1, 2]

df = df.dropna()

df["status"] = pd.cut(
    df["final_score"],
    bins=bins,
    labels=labels,
    include_lowest=True
)
st.title("Students Data")
st.subheader("Data")
st.dataframe(df.tail(10))

from sklearn.preprocessing import LabelEncoder

X = df[["jee_main_score", "jee_advanced_score", "mock_test_score_avg"]]
y = df["status"]
X = X.dropna()
y = y.loc[X.index]
y = y.astype(float)
le = LabelEncoder()
y_encoded = le.fit_transform(df['status'])

# -------------------- Model Selection --------------------
model_choice = st.selectbox("Choose a model", ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "DecisionTreeClassifier"])
if model_choice == "Logistic Regression":
    model = LogisticRegression(class_weight='balanced', max_iter=2000)
elif model_choice == "Random Forest":
    model = RandomForestClassifier(class_weight='balanced', random_state=42)
elif model_choice == "XGBoost":
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1.5)
elif model_choice == "LightGBM":
    model = LGBMClassifier(class_weight='balanced')
elif model_choice == "DecisionTreeClassifier":
    model = DecisionTreeClassifier(class_weight ='balanced')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

model.fit(X_train_scaled, y_train_resampled)

y_pred = model.predict(X_test_scaled)

st.subheader("Model Performance")
st.write("Accuracy:", accuracy_score(y_test, y_pred))
st.text("Classification Report:\n" + classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
st.pyplot(plt)

import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X.columns,
    class_names=le.classes_,
    mode='classification'
)


# 🎯 User Input Form
st.subheader("Predict Student Category")
st.markdown("Enter the scores below:")

jee_main_input = st.number_input("JEE Main Score", min_value=0, max_value=100, value=50)
jee_adv_input = st.number_input("JEE Advanced Score", min_value=0, max_value=100, value=50)
mock_avg_input = st.number_input("Mock Test Average Score", min_value=0, max_value=100, value=60)

# 🎯 Predict Button
if st.button("Predict Performance"):
    # Combine inputs into a DataFrame
    user_data = pd.DataFrame({
        'jee_main_score': [jee_main_input],
        'jee_advanced_score': [jee_adv_input],
        'mock_test_score_avg': [mock_avg_input]
    })

    # Scale the user input
    user_scaled = scaler.transform(user_data)

    # Make prediction
    prediction = model.predict(user_scaled)

    # Convert numeric label back to readable label
    # label_map = {0: "At Risk", 1: "Average", 2: "High Performer"}
    # prediction_label = label_map.get(prediction, "Unknown")

    prediction_label = le.inverse_transform(prediction)[0]
    # Display result
    # st.success(f"Prediction: **{prediction_label}**")

    st.subheader("Prediction")
    st.write(f"🎯 Predicted Category: **{prediction_label}**")

    # LIME Explanation
    exp = explainer.explain_instance(user_scaled[0], model.predict_proba, num_features=3)
    st.subheader("🔍 LIME Explanation")
    components.html(exp.as_html(), height=500)