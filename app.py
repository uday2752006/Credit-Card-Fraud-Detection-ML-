import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Set page configuration
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# Title and description
st.title("Credit Card Fraud Detection Dashboard")
st.markdown("""
This dashboard allows you to predict whether a credit card transaction is fraudulent using a pre-trained XGBoost model.
Enter transaction details below or upload a CSV file to get predictions.
""")

# Load the model
@st.cache_resource
def load_model():
    with open('project1.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Initialize scaler for Time and Amount
scaler = StandardScaler()
# Fit scaler on some sample data (replace with actual training data stats if available)
sample_data = pd.DataFrame({
    'Time': [0, 1, 2, 3, 4],
    'Amount': [149.62, 2.69, 378.66, 123.50, 69.99]
})
scaler.fit(sample_data[['Time', 'Amount']])

# Sidebar for input
st.sidebar.header("Input Transaction Details")
def get_user_input():
    input_data = {}
    input_data['Time'] = st.sidebar.number_input("Time (seconds since first transaction)", min_value=0.0, value=0.0)
    for i in range(1, 29):
        input_data[f'V{i}'] = st.sidebar.number_input(f'V{i} (PCA Feature)', value=0.0)
    input_data['Amount'] = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=0.0)
    return pd.DataFrame([input_data])

# File upload for batch predictions
uploaded_file = st.sidebar.file_uploader("Upload CSV file for batch predictions", type=["csv"])

# Function to preprocess input
def preprocess_input(input_df):
    input_scaled = input_df.copy()
    input_scaled[['Time', 'Amount']] = scaler.transform(input_df[['Time', 'Amount']])
    return input_scaled

# Function to plot ROC curve
def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC Curve (AUC={roc_auc:.3f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))
    fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        showlegend=True,
        grid=dict(rows=1, columns=1)
    )
    return fig

# Main app logic
if uploaded_file is not None:
    st.subheader("Batch Prediction Results")
    data = pd.read_csv(uploaded_file)
    if set(['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]).issubset(data.columns):
        X = data[['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']]
        X_scaled = preprocess_input(X)
        predictions = model.predict(X_scaled)
        data['Prediction'] = predictions
        data['Prediction'] = data['Prediction'].map({0: 'Normal', 1: 'Fraud'})
        
        st.write("Predictions:")
        st.dataframe(data[['Time', 'Amount', 'Prediction']])
        
        # If true labels are provided, show metrics
        if 'Class' in data.columns:
            y_true = data['Class']
            y_pred = predictions
            st.subheader("Model Performance")
            st.text("Classification Report:\n" + classification_report(y_true, y_pred, zero_division=0))
            cm = confusion_matrix(y_true, y_pred)
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm, x=['Normal', 'Fraud'], y=['Normal', 'Fraud'], colorscale='Blues',
                text=cm, texttemplate="%{text}", showscale=True
            ))
            fig_cm.update_layout(title="Confusion Matrix")
            st.plotly_chart(fig_cm)
            
            # ROC Curve
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]
            st.plotly_chart(plot_roc_curve(y_true, y_pred_proba))
    else:
        st.error("CSV must contain columns: Time, V1-V28, Amount")

else:
    st.subheader("Single Transaction Prediction")
    input_df = get_user_input()
    if st.sidebar.button("Predict"):
        input_scaled = preprocess_input(input_df)
        prediction = model.predict(input_scaled)[0]
        result = "Fraud" if prediction == 1 else "Normal"
        st.write(f"**Prediction**: This transaction is **{result}**")
        
        # Display probability
        proba = model.predict_proba(input_scaled)[0]
        st.write(f"**Probability of Fraud**: {proba[1]:.3f}")
        
        # Simple visualization
        fig = go.Figure(data=[
            go.Bar(x=['Normal', 'Fraud'], y=proba, marker_color=['green', 'red'])
        ])
        fig.update_layout(title="Prediction Probabilities", yaxis_title="Probability")
        st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and Plotly | Model: XGBoost | Data: Kaggle Credit Card Dataset")