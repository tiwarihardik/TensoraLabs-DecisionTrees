import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.title('ThinkForge - Decision Trees')
st.write('Where ideas are built.')

# Session state initialization
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_columns' not in st.session_state:
    st.session_state.X_columns = None
if 'target_categories' not in st.session_state:
    st.session_state.target_categories = None

# Upload CSV
file = st.file_uploader("Upload a CSV File", type='.csv')
if file:
    df = pd.read_csv(file).dropna()
    st.write("Data Preview:", df.head())

    target = st.selectbox('Select column to predict:', df.columns)
    features = st.multiselect('Select features to use:', df.columns.drop(target))
    depth = st.slider('Tree depth:', 1, 10, 3)

    if st.button('Train Model') and features:
        X = df[features]
        y = df[target]

        # Encode categorical variables if needed
        X_enc = pd.get_dummies(X)
        X_columns = X_enc.columns.tolist()  # Store the columns for later use
        st.session_state.X_columns = X_columns

        if y.dtype == 'object':
            target_categories = y.unique().tolist()
            y = pd.factorize(y)[0]
            st.session_state.target_categories = target_categories

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.3, random_state=42)
        model = DecisionTreeClassifier(max_depth=depth)
        model.fit(X_train, y_train)
        st.session_state.model = model

        acc = accuracy_score(y_test, model.predict(X_test))
        st.success(f"Model Accuracy: {acc:.2f}")

        st.subheader("Decision Tree")
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(model, filled=True, feature_names=X_enc.columns)
        st.pyplot(fig)

        st.subheader("Feature Importance")
        importance = pd.DataFrame({'Feature': X_enc.columns, 'Importance': model.feature_importances_})
        st.bar_chart(importance.set_index('Feature').sort_values('Importance', ascending=False))

# Prediction part, only runs when model is trained
if st.session_state.model:
    st.header("Make Predictions")

    # Simple input fields for each feature (numeric or categorical)
    user_input = {}
    for feature in features:
        if pd.api.types.is_numeric_dtype(df[feature]):
            user_input[feature] = st.number_input(f"{feature}:")
        else:
            user_input[feature] = st.selectbox(f"Select {feature}:", df[feature].unique())

    # Predict on button click
    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        input_enc = pd.get_dummies(input_df).reindex(columns=st.session_state.X_columns, fill_value=0)
        pred = st.session_state.model.predict(input_enc)[0]
        label = st.session_state.target_categories[pred] if st.session_state.target_categories else pred
        st.success(f"Prediction: {label}")
