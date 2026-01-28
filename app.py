import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Page configuration
st.set_page_config(page_title="Model Evaluator", layout="wide")

st.title("📊 Model Evaluation Dashboard")
st.write("Upload your test dataset (CSV) to evaluate model performance.")

# --- a. Dataset Upload Option ---
uploaded_file = st.sidebar.file_uploader("Upload Test CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True)
    st.write("### Preview of Uploaded Test Data")
    st.dataframe(df.head())

    # Sidebar selection for Target and Features
    target_col = st.sidebar.selectbox("Select Target Variable (Y)", df.columns)
    feature_cols = st.sidebar.multiselect("Select Feature Variables (X)", [c for c in df.columns if c != target_col])

    if feature_cols:
        X_test = df[feature_cols]
        y_test = df[target_col]

        # --- b. Model Selection Dropdown ---
        model_choice = st.sidebar.selectbox(
            "Select Model to Evaluate", 
            ["Logistic Regression", "Random Forest"]
        )

        # Initialize and "Mock Train" (In a real scenario, you'd load a saved .pkl file)
        # We are fitting here quickly for demonstration purposes.
        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        else:
            model = RandomForestClassifier()

        model.fit(X_test, y_test) # Fitting on test data just to generate metrics for the UI
        y_pred = model.predict(X_test)

        # Layout Columns
        col1, col2 = st.columns(2)

        with col1:
            # --- c. Display of Evaluation Metrics ---
            st.subheader("📈 Evaluation Metrics")
            acc = accuracy_score(y_test, y_pred)
            st.metric(label="Accuracy Score", value=f"{acc:.2%}")
            
            st.write("**Classification Report:**")
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            st.table(pd.DataFrame(report_dict).transpose())

        with col2:
            # --- d. Confusion Matrix ---
            st.subheader("🧩 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted labels')
            ax.set_ylabel('True labels')
            st.pyplot(fig)

else:
    st.info("Please upload a CSV file via the sidebar to begin.")