
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Smart Energy Consumption",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Smart Energy Consumption Analysis and Prediction")
st.markdown(
    "### Machine Learning Based Household Electricity Consumption Analysis"
)

st.write(
    "This application analyzes household electricity consumption "
    "and predicts future consumption using Linear Regression "
    "and Random Forest Regression."
)

st.divider()

# ----------------------------------
# LOAD DATA AND MODELS
# ----------------------------------

@st.cache_data
def load_data():
    path = "dataset/household_power_consumption.txt"

    df = pd.read_csv(
        path,
        sep=";",
        na_values=["?"],
        low_memory=False
    )

    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        dayfirst=True,
        errors="coerce"
    )

    df["Global_active_power"] = pd.to_numeric(
        df["Global_active_power"],
        errors="coerce"
    )

    df.dropna(
        subset=["datetime", "Global_active_power"],
        inplace=True
    )

    df = df.sort_values("datetime")

    df["energy_kwh"] = df["Global_active_power"] / 60

    return df


@st.cache_resource
def load_models():
    linear = joblib.load("models/linear_regression.pkl")
    forest = joblib.load("models/random_forest.pkl")

    return linear, forest


try:
    df = load_data()
    linear_model, forest_model = load_models()

except Exception as e:
    st.error(f"Unable to load project files: {e}")
    st.stop()


# ----------------------------------
# SIDEBAR
# ----------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a section",
    [
        "Dashboard",
        "Data Analysis",
        "Energy Prediction",
        "Model Comparison",
        "About Project"
    ]
)


# ----------------------------------
# DASHBOARD
# ----------------------------------

if page == "Dashboard":

    st.header("Energy Consumption Overview")

    total_records = len(df)

    avg_power = df["Global_active_power"].mean()

    total_energy = df["energy_kwh"].sum()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Records",
        f"{total_records:,}"
    )

    col2.metric(
        "Average Power",
        f"{avg_power:.3f} kW"
    )

    col3.metric(
        "Estimated Energy",
        f"{total_energy:,.1f} kWh"
    )

    st.subheader("Hourly Energy Consumption")

    hourly = (
        df.set_index("datetime")["Global_active_power"]
        .resample("h")
        .mean()
        .dropna()
    )

    hourly_energy = hourly  # Average kW for each hour

    st.line_chart(
        hourly_energy,
        height=350
    )

    st.subheader("Daily Average Power Consumption")

    daily = (
        df.set_index("datetime")["Global_active_power"]
        .resample("D")
        .mean()
        .dropna()
    )

    st.bar_chart(daily, height=300)


# ----------------------------------
# DATA ANALYSIS
# ----------------------------------

elif page == "Data Analysis":

    st.header("Exploratory Data Analysis")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.subheader("Power Consumption Distribution")

    fig, ax = plt.subplots()

    ax.hist(
        df["Global_active_power"],
        bins=40
    )

    ax.set_xlabel("Global Active Power (kW)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Household Power Consumption")

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Daily Consumption Trend")

    daily = (
        df.set_index("datetime")["Global_active_power"]
        .resample("D")
        .mean()
        .dropna()
    )

    st.line_chart(daily)


# ----------------------------------
# ENERGY PREDICTION
# ----------------------------------

elif page == "Energy Prediction":

    st.header("Predict Household Energy Consumption")

    st.write(
        "Enter the previous hourly consumption values "
        "to predict the next hour."
    )

    col1, col2 = st.columns(2)

    with col1:

        prediction_date = st.date_input(
            "Prediction Date"
        )

        prediction_hour = st.slider(
            "Prediction Hour",
            0,
            23,
            12
        )

        lag_1 = st.number_input(
            "Previous Hour Consumption (kWh)",
            min_value=0.0,
            value=1.0
        )

        lag_2 = st.number_input(
            "Consumption 2 Hours Ago (kWh)",
            min_value=0.0,
            value=1.0
        )

    with col2:

        lag_3 = st.number_input(
            "Consumption 3 Hours Ago (kWh)",
            min_value=0.0,
            value=1.0
        )

        rolling_mean = st.number_input(
            "Average of Previous 3 Hours (kWh)",
            min_value=0.0,
            value=1.0
        )

        model_choice = st.selectbox(
            "Select ML Model",
            ["Linear Regression", "Random Forest"]
        )

    if st.button("Predict Energy Consumption"):

        features = pd.DataFrame([{
            "hour": prediction_hour,
            "day": prediction_date.day,
            "month": prediction_date.month,
            "weekday": prediction_date.weekday(),
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3,
            "rolling_mean_3": rolling_mean
        }])

        if model_choice == "Linear Regression":
            prediction = linear_model.predict(features)[0]
        else:
            prediction = forest_model.predict(features)[0]

        prediction = max(0, prediction)

        st.success(
            f"Predicted Energy Consumption: {prediction:.3f} kWh"
        )

        st.info(
            "This is an estimated next-hour consumption value."
        )

        tariff = st.number_input(
            "Electricity tariff (currency per kWh)",
            min_value=0.0,
            value=8.0
        )

        st.metric(
            "Estimated Cost for One Hour",
            f"{prediction * tariff:.2f}"
        )


# ----------------------------------
# MODEL COMPARISON
# ----------------------------------

elif page == "Model Comparison":

    st.header("Machine Learning Model Comparison")

    results_path = "results/model_comparison.csv"

    if os.path.exists(results_path):

        results = pd.read_csv(results_path)

        st.subheader("Evaluation Metrics")

        st.dataframe(
            results.style.format({
                "MAE": "{:.4f}",
                "MSE": "{:.4f}",
                "RMSE": "{:.4f}",
                "R2 Score": "{:.4f}"
            }),
            use_container_width=True
        )

        st.subheader("MAE Comparison")

        st.bar_chart(
            results.set_index("Model")[["MAE"]]
        )

        st.subheader("RMSE Comparison")

        st.bar_chart(
            results.set_index("Model")[["RMSE"]]
        )

        st.subheader("R² Score Comparison")

        st.bar_chart(
            results.set_index("Model")[["R2 Score"]]
        )

        best_model = results.loc[
            results["RMSE"].idxmin(),
            "Model"
        ]

        st.success(
            f"Best Performing Model: {best_model}"
        )

        # Actual vs Predicted Consumption Graph

        st.divider()
        st.subheader("Actual vs Predicted Energy Consumption")

        prediction_path = "results/predictions.csv"

        if os.path.exists(prediction_path):

            prediction_data = pd.read_csv(prediction_path)

            st.write(
                "Comparison of actual electricity consumption "
                "with predictions from both machine learning models."
            )

            # Display a sample of 200 test observations
            graph_data = prediction_data[
                ["Actual", "Linear Regression", "Random Forest"]
            ].head(200)

            st.line_chart(
                graph_data,
                height=400
            )

            st.subheader("Prediction Data Preview")

            st.dataframe(
                prediction_data.head(10),
                use_container_width=True
            )

        else:
            st.warning("Prediction results file not found.")

    else:

        st.warning("Model comparison results not found.")


# ----------------------------------
# ABOUT PROJECT
# ----------------------------------

elif page == "About Project":

    st.header("About the Project")

    st.write("""
    **Project Title:** Smart Energy Consumption Analysis and
    Prediction Using Machine Learning

    **Machine Learning Algorithms:**
    - Linear Regression
    - Random Forest Regression

    **Dataset:** UCI Individual Household Electric Power
    Consumption Dataset

    **Technologies Used:**
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Matplotlib
    - Streamlit

    **Evaluation Metrics:**
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - R² Score
    """)

    st.info(
        "Developed as a college mini-project."
    )

st.divider()

st.caption(
    "Smart Energy Consumption Analysis and Prediction "
    "| Machine Learning Mini Project"
)
