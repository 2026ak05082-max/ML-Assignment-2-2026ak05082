from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

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

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
TEST_DATA_FILE = BASE_DIR / "test_data.csv"
METRICS_FILE = BASE_DIR / "model_metrics.csv"

# Page configuration
st.set_page_config(
    page_title="ML Assignment 2",
    page_icon="🔬",
    layout="wide"
)

st.title("Machine Learning Assignment 2")
st.subheader("Breast Cancer Wisconsin Diagnostic Classification")
st.write(
    "Select a trained machine-learning model and upload "
    "test_data.csv to evaluate model performance."
)
st.warning(
    "Educational project only. This application is not "
    "intended for medical diagnosis."
)

# Saved model files
model_files = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Decision Tree": MODEL_DIR / "decision_tree.joblib",
    "K-Nearest Neighbors": MODEL_DIR / "knn.joblib",
    "Gaussian Naive Bayes": MODEL_DIR / "naive_bayes.joblib",
    "Support Vector Machine": MODEL_DIR / "svm.joblib"
}

# Sidebar
st.sidebar.header("Model Evaluation")
selected_model = st.sidebar.selectbox(
    "Select Model",
    list(model_files.keys())
)
uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

# Sample test-data download
if TEST_DATA_FILE.exists():
    st.sidebar.download_button(
        label="Download Sample Test Data",
        data=TEST_DATA_FILE.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv"
    )

# Process uploaded data
if uploaded_file is not None:
    try:
        test_data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Test Dataset")
        col1, col2 = st.columns(2)
        col1.metric("Rows", test_data.shape[0])
        col2.metric("Columns", test_data.shape[1])
        st.dataframe(test_data.head(10), use_container_width=True)

        if "target" not in test_data.columns:
            st.error("Uploaded CSV must contain a column named 'target'.")
            st.stop()

        X_test = test_data.drop(columns=["target"])
        y_test = pd.to_numeric(test_data["target"], errors="raise").astype(int)

        if not set(y_test.unique()).issubset({0, 1}):
            st.error("Target column must contain only 0 and 1.")
            st.stop()

        if y_test.nunique() < 2:
            st.error("The uploaded data must contain both target classes for AUC.")
            st.stop()

        X_test = X_test.apply(pd.to_numeric, errors="raise")

        if X_test.isna().any().any() or y_test.isna().any():
            st.error("Missing values are not supported.")
            st.stop()

        model_path = model_files[selected_model]

        if not model_path.exists():
            st.error(f"Model file not found: {model_path}")
            st.stop()

        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.success(f"Model evaluated successfully: {selected_model}")

        st.subheader("Evaluation Metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{accuracy:.4f}")
        c2.metric("AUC", f"{auc:.4f}")
        c3.metric("Precision", f"{precision:.4f}")

        c4, c5, c6 = st.columns(3)
        c4.metric("Recall", f"{recall:.4f}")
        c5.metric("F1 Score", f"{f1:.4f}")
        c6.metric("MCC", f"{mcc:.4f}")

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Malignant", "Actual Benign"],
            columns=["Predicted Malignant", "Predicted Benign"]
        )
        st.dataframe(cm_df, use_container_width=True)

        st.subheader("Classification Report")
        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Malignant", "Benign"],
            output_dict=True,
            zero_division=0
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

        st.subheader("Prediction Results")
        prediction_results = test_data.copy()
        prediction_results["prediction"] = y_pred
        prediction_results["prediction_label"] = [
            "Benign" if value == 1 else "Malignant"
            for value in y_pred
        ]
        prediction_results["benign_probability"] = y_prob
        st.dataframe(prediction_results.head(20), use_container_width=True)

        csv_output = prediction_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Prediction Results",
            data=csv_output,
            file_name="predictions.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error(f"Unable to evaluate uploaded data: {error}")
else:
    st.info("Please upload test_data.csv from the sidebar.")

    if METRICS_FILE.exists():
        st.subheader("Saved Model Performance Comparison")
        metrics_df = pd.read_csv(METRICS_FILE)
        st.dataframe(metrics_df, use_container_width=True)

# Project information
st.divider()
st.subheader("Models Implemented")
st.markdown(
    """
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors
4. Gaussian Naive Bayes
5. Support Vector Machine
"""
)

st.subheader("Target Classes")
st.markdown(
    """
- **0 = Malignant**
- **1 = Benign**
"""
)

st.caption(
    "Machine Learning Assignment 2 | "
    "Breast Cancer Wisconsin Diagnostic Dataset"
)
