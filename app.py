import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Sales Predictor", layout="wide", initial_sidebar_state="expanded")

# ---------------- SESSION STATE ----------------
if "single_prediction" not in st.session_state:
    st.session_state.single_prediction = None

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("sales_model.pkl", "rb"))

# ---------------- TITLE ----------------
st.title("📊 Sales Prediction App")
st.markdown("Predict Sales based on advertising spend in *TV, **Radio, and **Newspaper*.")

predicted_sales=None
# ---------------- SINGLE PREDICTION ----------------
st.sidebar.header("🛠 Enter Ad Spend Details")
tv = st.sidebar.number_input("💻 TV Ad Spend", min_value=0.0, max_value=300.0, value=150.0, step=0.5)
radio = st.sidebar.number_input("📻 Radio Ad Spend", min_value=0.0, max_value=50.0, value=25.0, step=0.5)
newspaper = st.sidebar.number_input("📰 Newspaper Ad Spend", min_value=0.0, max_value=100.0, value=20.0, step=0.5)

if st.sidebar.button("🔮 Predict Sales"):
    predicted_sales = model.predict([[tv, radio, newspaper]])
    st.session_state.single_prediction = predicted_sales[0]  # ✅ Store in session
    st.subheader("📈 Predicted Sales for Single Input:")
    st.success(f"{predicted_sales[0]:.2f} units")

# ---------------- BULK PREDICTION ----------------
st.markdown("---")
st.header("📂 Bulk Prediction from CSV")
if predicted_sales is None:
    st.info("⚠ Tip: Perform a single prediction first to enable the combined chart view.")

uploaded_file = st.file_uploader("Upload a CSV file with columns: TV, Radio, Newspaper", type=["csv"], key="bulk_csv")

# Auto-create a sample file if not uploaded (for demo)
if uploaded_file is None and not os.path.exists("test_sales.csv"):
    sample_data = pd.DataFrame({
        "TV": [100, 200, 150],
        "Radio": [20, 30, 40],
        "Newspaper": [10, 20, 15]
    })
    sample_data.to_csv("test_sales.csv", index=False)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    data = pd.read_csv("test_sales.csv")
    st.info("📌 Using auto-generated sample dataset (test_sales.csv).")

required_cols = {"TV", "Radio", "Newspaper"}
if required_cols.issubset(data.columns):
    if st.button("🔮 Predict Bulk Sales"):
        predictions = model.predict(data[["TV", "Radio", "Newspaper"]])
        data["Predicted_Sales"] = predictions

        st.subheader("📋 Predictions Table:")
        st.dataframe(data)

        # ✅ Bulk Predictions Chart
        labels = [f"Row {i}" for i in range(len(data))]
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=data['Predicted_Sales'],
                marker_color='skyblue',
                text=data['Predicted_Sales'],
                textposition='auto',
                hovertemplate="<b>%{x}</b><br>Predicted Sales: %{y:.2f}<extra></extra>"
            )
        ])
        fig.update_layout(
            xaxis_title="Campaigns",
            yaxis_title="Predicted Sales",
            title="📈 Bulk Sales Predictions",
            bargap=0.3,
            plot_bgcolor='rgba(0,0,0,0)',
            width=800,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # ✅ Download CSV
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name='bulk_predictions.csv',
            mime='text/csv'
        )

        # ✅ Combined Chart (Single vs Bulk)
        if st.session_state.single_prediction is not None:
            st.subheader("📊 Interactive Comparison: Single vs Bulk Predictions")
            combined_df = data.copy()
            single_row = pd.DataFrame({
                "TV": [tv],
                "Radio": [radio],
                "Newspaper": [newspaper],
                "Predicted_Sales": [st.session_state.single_prediction]
            })
            combined_df = pd.concat([combined_df, single_row], ignore_index=True)
            labels = [f"Row {i}" for i in range(len(data))] + ["Single Prediction"]
            colors = ['skyblue'] * len(data) + ['salmon']

            fig = go.Figure(data=[
                go.Bar(
                    x=labels,
                    y=combined_df['Predicted_Sales'],
                    marker_color=colors,
                    text=combined_df['Predicted_Sales'],
                    textposition='auto',
                    hovertemplate="<b>%{x}</b><br>Predicted Sales: %{y:.2f}<extra></extra>"
                )
            ])
            fig.update_layout(
                xaxis_title="Campaigns",
                yaxis_title="Predicted Sales",
                title="📈 Single Prediction vs Bulk Campaigns",
                bargap=0.3,
                plot_bgcolor='rgba(0,0,0,0)',
                width=800,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    st.error("❌ CSV must have columns: TV, Radio, Newspaper")
