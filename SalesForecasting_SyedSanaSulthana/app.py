# Sales Forecasting Dashboard
# Created by Syed Sana Sulthana


# Importing the required libraries
import streamlit as st
import pandas as pd
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# Dashboard Title
# -----------------------------
st.title("📈 End-to-End Sales Forecasting & Demand Intelligence System")

st.markdown("""
### XyLofy AI Data Science Internship

This dashboard presents the complete analysis of the Superstore Sales Dataset.

The project includes:

- 📊 Exploratory Data Analysis (EDA)
- 📈 Sales Forecasting using SARIMA, Prophet and XGBoost
- 🚨 Sales Anomaly Detection
- 📦 Product Demand Segmentation using K-Means Clustering

The objective of this project is to forecast future sales, detect unusual sales patterns, and provide useful business insights for inventory planning.
""")

st.markdown("---")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():

    # reading dataset
     BASE_DIR = os.path.dirname(os.path.abspath(__file__))

     csv_path = os.path.join(BASE_DIR, "train.csv")

     data = pd.read_csv(csv_path)

    # converting dates into datetime format
    data["Order Date"] = pd.to_datetime(
        data["Order Date"],
        dayfirst=True
    )

    data["Ship Date"] = pd.to_datetime(
        data["Ship Date"],
        dayfirst=True
    )

    return data

data = load_data()

# -----------------------------
# Function to Display Charts
# -----------------------------
def display_chart(filename, title):
    chart_path = os.path.join(BASE_DIR, "Charts", filename)

    if os.path.exists(chart_path):
        st.subheader(title)
        st.image(chart_path, use_container_width=True)
    else:
        st.warning(f"{filename} not found inside Charts folder.")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📊 Dashboard Menu")

st.sidebar.markdown("""
### Welcome!

This dashboard was developed as part of my
**XyLofy AI Data Science Internship**.

Use the menu below to explore different sections
of the project.
""")

page = st.sidebar.radio(
    "Navigate",
    (
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    )
)

st.sidebar.markdown("---")
st.sidebar.success("🏆 Best Model : XGBoost")
st.sidebar.info("📁 Dataset : Superstore Sales")

# ==========================================================
# PAGE 1 : SALES OVERVIEW
# ==========================================================

if page == "Sales Overview":

    st.header("📊 Sales Overview Dashboard")

    total_sales = data["Sales"].sum()

    total_orders = len(data)

    average_sales = data["Sales"].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Sales", f"${total_sales:,.2f}")

    col2.metric("Total Orders", total_orders)

    col3.metric("Average Sales", f"${average_sales:.2f}")

    st.markdown("---")

    display_chart(
        "monthly_sales_trend.png",
        "Monthly Sales Trend"
    )

    display_chart(
        "category_sales.png",
        "Sales by Category"
    )

    display_chart(
        "region_growth.png",
        "Regional Sales Growth"
    )

    st.success("The overall sales trend shows steady growth with noticeable seasonal spikes.")


# PAGE 2 : FORECAST EXPLORER


elif page == "Forecast Explorer":

    st.header("📈 Forecast Explorer")

    st.write("""
This page compares the forecasting models that I implemented
during the project. Based on the evaluation metrics,
XGBoost performed better than SARIMA and Prophet.
""")

    st.markdown("---")

    # Model Performance
    st.subheader("Model Performance Comparison")

    comparison = pd.DataFrame({

        "Model":["SARIMA","Prophet","XGBoost"],

        "MAE":[18031.40,40970.33,14148.40],

        "RMSE":[19009.18,53868.95,16213.61],

        "MAPE":[18.97,40.04,14.34]

    })

    st.dataframe(comparison, use_container_width=True)

    st.success("XGBoost achieved the lowest MAE, RMSE and MAPE, so it was selected as the best forecasting model.")

    st.markdown("---")

    # Forecast Horizon
    months = st.slider(
        "Select Forecast Horizon (Months)",
        1,
        3,
        3
    )

    st.subheader("Forecast by Category / Region")

    option = st.selectbox(

        "Select Category or Region",

        (

            "Furniture",

            "Technology",

            "Office Supplies",

            "West Region",

            "East Region"

        )

    )

    # Forecast values from XGBoost

    forecast_values = {

        "Furniture":[23781.121,35154.785,36476.120],

        "Technology":[23677.203,26478.621,21074.945],

        "Office Supplies":[27748.530,27748.530,27748.530],

        "West Region":[16328.995,19269.371,23942.812],

        "East Region":[23228.467,23284.807,25352.947]

    }

    values = forecast_values[option]

    forecast_df = pd.DataFrame({

        "Month":[

            "Month 1",

            "Month 2",

            "Month 3"

        ],

        "Predicted Sales":values

    })

    st.table(forecast_df.head(months))

    st.markdown("---")

    st.subheader("Forecast Charts")

    display_chart(

        "sarima_forecast.png",

        "SARIMA Forecast"

    )

    display_chart(

        "prophet_forecast.png",

        "Prophet Forecast"

    )

    display_chart(

        "prophet_components.png",

        "Prophet Trend & Seasonality"

    )

    display_chart(

        "xgboost_forecast.png",

        "XGBoost Forecast"

    )

    display_chart(

        "category_region_forecast.png",

        "Category & Region Forecast"

    )

    st.info("""
Observation:

• Prophet produced the highest forecasting error.

• SARIMA gave reasonable predictions.

• XGBoost produced the most accurate forecasts with the lowest error values.

Therefore, XGBoost was selected as the final forecasting model for this project.
""")
    

# PAGE 3 : ANOMALY REPORT


elif page == "Anomaly Report":

    st.header("🚨 Sales Anomaly Detection")

    st.write("""
In this task, I used two different methods to detect unusual sales patterns.

1. Isolation Forest
2. Z-Score Method

These methods help identify weeks where sales were unusually high or low.
""")

    st.markdown("---")

    # Summary

    col1, col2 = st.columns(2)

    col1.metric(
        "Isolation Forest",
        "11 Anomalies"
    )

    col2.metric(
        "Z-Score Method",
        "11 Anomalies"
    )

    st.success("Both methods detected 11 unusual sales weeks.")

    st.markdown("---")

    # Isolation Forest Chart

    display_chart(
        "isolation_forest_anomalies.png",
        "Isolation Forest Anomaly Detection"
    )

    # Z Score Chart

    display_chart(
        "zscore_anomalies.png",
        "Z-Score Anomaly Detection"
    )

    st.markdown("---")

    st.subheader("Business Interpretation")

    anomaly_table = pd.DataFrame({

        "Possible Reason":[

            "Festival Season",

            "Heavy Discounts",

            "Large Bulk Orders",

            "Stock Shortage",

            "Unexpected Demand"

        ],

        "Impact":[

            "Sales Spike",

            "Higher Revenue",

            "Large Orders",

            "Sales Drop",

            "Demand Variation"

        ]

    })

    st.table(anomaly_table)

    st.markdown("---")

    st.subheader("Observation")

    st.write("""

• Isolation Forest and Z-Score detected almost the same abnormal weeks.

• Most of the unusual sales were likely caused by festive seasons and discount offers.

• Some low-sales weeks may indicate inventory shortages or reduced customer demand.

• Detecting anomalies helps businesses plan inventory and avoid unexpected losses.

""")
    

# PAGE 4 : PRODUCT DEMAND SEGMENTS


elif page == "Product Demand Segments":

    st.header("📦 Product Demand Segmentation")

    st.write("""
In this task, I grouped product sub-categories based on their
sales behaviour using K-Means Clustering.

The Elbow Method was used to determine the optimal number of clusters.
""")

    st.markdown("---")

    # Elbow Method Chart
    display_chart(
        "elbow_method.png",
        "Elbow Method"
    )

    # Product Cluster Chart
    display_chart(
        "product_clusters.png",
        "Product Demand Clusters"
    )

    st.markdown("---")

    st.subheader("Cluster Interpretation")

    cluster_table = pd.DataFrame({

        "Cluster":[
            "Cluster 0",
            "Cluster 1",
            "Cluster 2",
            "Cluster 3"
        ],

        "Meaning":[
            "High Volume, Stable Demand",
            "Growing Demand",
            "Low Volume, High Volatility",
            "Declining Demand"
        ],

        "Recommended Strategy":[
            "Maintain sufficient inventory",
            "Increase stock gradually",
            "Monitor demand carefully",
            "Reduce inventory and avoid overstocking"
        ]

    })

    st.table(cluster_table)

    st.markdown("---")

    st.subheader("Project Summary")

    st.success("""
This project successfully analysed historical sales data,
built multiple forecasting models, detected anomalies,
and segmented products based on demand.

Among all forecasting models,
XGBoost achieved the best performance and was selected
as the final model for future sales prediction.
""")
    
st.markdown("---")

st.caption(
    "Developed by Syed Sana Sulthana | XyLofy AI Data Science Internship | 2026"
)
