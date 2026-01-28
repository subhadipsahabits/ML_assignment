# ==========================================
# Streamlit App: Binary Classification Models
# (With NaN Handling)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

# ------------------------------------------
# Streamlit UI
# ------------------------------------------
st.set_page_config(page_title="Binary Classification App", layout="wide")

st.title("📊 Binary Classification Model Evaluation")

# ------------------------------------------
# Dataset Upload
# ------------------------------------------
uploaded_file = st.file_uploader(
    "Upload CSV file (test-sized dataset only)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    df.drop(columns=['employee_id'],inplace=True)


    target_column = "attrition"   
    # Drop NaN in target column ONLY
    df = df.dropna(subset=[target_column])

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if y.dtype == "object":
        y = y.map({"Yes": 1, "No": 0})

    # ------------------------------------------
    # Feature Processing
    # ------------------------------------------
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )

    # ------------------------------------------
    # Train-Test Split
    # ------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ------------------------------------------
    # Model Selection
    # ------------------------------------------
    model_name = st.selectbox(
        "Select Machine Learning Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Naive Bayes (Gaussian)",
            "Random Forest",
            "XGBoost"
        ]
    )

    # ------------------------------------------
    # Model Definitions
    # ------------------------------------------
    model_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        )
    }

    if st.button("🚀 Train & Evaluate Model"):
        model = model_dict[model_name]

        # ------------------------------------------
        # Training
        # ------------------------------------------
        if model_name == "Naive Bayes (Gaussian)":
            X_train_prep = preprocessor.fit_transform(X_train).toarray()
            X_test_prep = preprocessor.transform(X_test).toarray()

            model.fit(X_train_prep, y_train)
            y_pred = model.predict(X_test_prep)
            y_prob = model.predict_proba(X_test_prep)[:, 1]
        else:
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", model)
                ]
            )

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

        # ------------------------------------------
        # Metrics
        # ------------------------------------------
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader("📈 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.4f}")
        col1.metric("AUC", f"{auc:.4f}")

        col2.metric("Precision", f"{precision:.4f}")
        col2.metric("Recall", f"{recall:.4f}")

        col3.metric("F1 Score", f"{f1:.4f}")
        col3.metric("MCC", f"{mcc:.4f}")

        # ------------------------------------------
        # Confusion Matrix
        # ------------------------------------------
        st.subheader("🔍 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        st.dataframe(
            pd.DataFrame(
                cm,
                columns=["Predicted 0", "Predicted 1"],
                index=["Actual 0", "Actual 1"]
            )
        )

        # ------------------------------------------
        # Classification Report
        # ------------------------------------------
        st.subheader("📄 Classification Report")
        st.text(classification_report(y_test, y_pred))

else:
    st.info("Please upload a CSV file to begin.")
