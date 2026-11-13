# ========================= PART 1: IMPORTS, CONFIGURATION,
#                         DATASET AND MODEL LOADING =========================

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ---------------------------- Page configuration ----------------------------

st.set_page_config(
    page_title="Smart Energy Consumption",
    page_icon="⚡",
    layout="wide",
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

# ---------------------------- Project paths ----------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
DATA_PATH = DATA_DIR / "household_power_consumption.txt"

LINEAR_MODEL_PATH = BASE_DIR / "models" / "linear_regression.pkl"
FOREST_MODEL_PATH = BASE_DIR / "models" / "random_forest.pkl"

RESULTS_DIR = BASE_DIR / "results"
MODEL_RESULTS_PATH = RESULTS_DIR / "model_comparison.csv"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)

FEATURES = [
    "hour",
    "day",
    "month",
    "weekday",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
]


# ---------------------------- Dataset download -------------------------------

def ensure_dataset_exists():
    """Download and extract the UCI dataset if the text file is absent."""

    if DATA_PATH.exists():
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "household_power_consumption.zip"

    try:
        with st.spinner("Downloading the household electricity dataset..."):
            request = urllib.request.Request(
                DATASET_URL,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                with open(zip_path, "wb") as output_file:
                    shutil.copyfileobj(response, output_file)

            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(DATA_DIR)

            extracted_file = next(
                DATA_DIR.rglob("household_power_consumption.txt"),
                None,
            )

            if extracted_file is None:
                raise FileNotFoundError(
                    "The expected household_power_consumption.txt "
                    "file was not found inside the UCI ZIP archive."
                )

            if extracted_file.resolve() != DATA_PATH.resolve():
                shutil.move(str(extracted_file), str(DATA_PATH))

            if zip_path.exists():
                zip_path.unlink()

    except Exception as exc:
        st.error(
            "The electricity dataset could not be downloaded. "
            "Check the internet connection or place "
            "'household_power_consumption.txt' inside the dataset folder."
        )
        st.exception(exc)
        st.stop()


# ---------------------------- Load and clean data ----------------------------

@st.cache_data(show_spinner=False)
def load_data():
    """Read, clean and index the original UCI household power dataset."""

    ensure_dataset_exists()

    data = pd.read_csv(
        DATA_PATH,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )

    required_columns = {"Date", "Time", "Global_active_power"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    data["datetime"] = pd.to_datetime(
        data["Date"].astype(str) + " " + data["Time"].astype(str),
        dayfirst=True,
        errors="coerce",
    )

    data["Global_active_power"] = pd.to_numeric(
        data["Global_active_power"],
        errors="coerce",
    )

    data = data.dropna(
        subset=["datetime", "Global_active_power"]
    ).copy()

    data = data.sort_values("datetime")
    data["energy_kwh"] = data["Global_active_power"]

    # Keep datetime as both an index and a column. This supports the
    # existing dashboard sections that refer to either representation.
    data = data.set_index("datetime", drop=False)

    return data


# ---------------------------- Load saved ML models ---------------------------

@st.cache_resource
def load_models():
    """Load the two models previously trained by train_model.py."""

    if not LINEAR_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {LINEAR_MODEL_PATH}"
        )

    if not FOREST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {FOREST_MODEL_PATH}"
        )

    linear_model = joblib.load(LINEAR_MODEL_PATH)
    forest_model = joblib.load(FOREST_MODEL_PATH)

    return linear_model, forest_model


try:
    df = load_data()
    linear_model, forest_model = load_models()
except Exception as error:
    st.error("Unable to load the project data or trained models.")
    st.exception(error)
    st.stop()


# ---------------------------- Common data helpers ----------------------------

def get_hourly_data():
    """Return hourly mean active power, matching the training script."""

    hourly = (
        df["Global_active_power"]
        .resample("h")
        .mean()
        .dropna()
    )
    hourly.name = "energy_kwh"
    return hourly


def make_feature_row(timestamp, lag_1, lag_2, lag_3):
    """Build features in exactly the order/names used during training."""

    rolling_mean = (float(lag_1) + float(lag_2) + float(lag_3)) / 3.0

    return pd.DataFrame(
        [
            {
                "hour": int(timestamp.hour),
                "day": int(timestamp.day),
                "month": int(timestamp.month),
                "weekday": int(timestamp.dayofweek),
                "lag_1": float(lag_1),
                "lag_2": float(lag_2),
                "lag_3": float(lag_3),
                "rolling_mean_3": float(rolling_mean),
            }
        ],
        columns=FEATURES,
    )


def predict_with_selected_model(model_name, feature_row):
    """Run prediction with the selected saved model."""

    if model_name == "Linear Regression":
        model = linear_model
    else:
        model = forest_model

    prediction = float(model.predict(feature_row)[0])
    return max(0.0, prediction)



# ========================= PART 2: SIDEBAR, DASHBOARD,
#                         DATA ANALYSIS =========================

# ---------------------------- Sidebar navigation -----------------------------

st.sidebar.title("Navigation")
st.sidebar.caption("Smart Energy Consumption System")

page = st.sidebar.radio(
    "Select a section",
    [
        "Dashboard",
        "Data Analysis",
        "Energy Prediction",
        "Model Comparison",
        "About Project",
    ],
)

st.sidebar.divider()
st.sidebar.info(
    "Machine Learning Models:\n"
    "- Linear Regression\n"
    "- Random Forest Regression"
)

# Prepare hourly data once for pages that need it.
hourly_data = get_hourly_data()


# ---------------------------- DASHBOARD --------------------------------------

if page == "Dashboard":

    st.header("Energy Consumption Dashboard")
    st.write(
        "An overview of household electricity consumption "
        "based on the historical dataset."
    )

    total_energy = float(hourly_data.sum())
    average_hourly = float(hourly_data.mean())
    maximum_hourly = float(hourly_data.max())
    total_records = int(len(df))

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric("Total hourly values (sum)", f"{total_energy:,.2f}")
    metric2.metric("Average hourly reading", f"{average_hourly:.3f}")
    metric3.metric("Maximum hourly reading", f"{maximum_hourly:.3f}")
    metric4.metric("Valid source records", f"{total_records:,}")

    st.caption(
        "The project training script uses the hourly mean of "
        "Global_active_power as its prediction target. The displayed "
        "energy_kwh label follows that original project convention."
    )

    st.subheader("Hourly Energy Consumption")
    st.line_chart(hourly_data, height=320)

    st.subheader("Monthly Average Power Consumption")
    monthly_average = (
        df["Global_active_power"]
        .resample("ME")
        .mean()
        .dropna()
    )
    st.bar_chart(monthly_average, height=320)

    st.subheader("Actual vs Predicted Values")

    if PREDICTIONS_PATH.exists():
        try:
            prediction_data = pd.read_csv(PREDICTIONS_PATH)

            if "Actual" in prediction_data.columns:
                chart_columns = [
                    col for col in
                    ["Actual", "Linear Regression", "Random Forest"]
                    if col in prediction_data.columns
                ]
                if len(chart_columns) >= 2:
                    st.line_chart(
                        prediction_data[chart_columns].head(500),
                        height=320,
                    )
                else:
                    st.info(
                        "The predictions file does not contain enough "
                        "columns to draw the comparison chart."
                    )
            else:
                st.info(
                    "The predictions file does not contain an 'Actual' column."
                )
        except Exception as error:
            st.warning(f"Could not display saved predictions: {error}")
    else:
        st.info(
            "Saved predictions.csv was not found. Train the models "
            "to generate the actual-versus-predicted chart."
        )


# ---------------------------- DATA ANALYSIS ----------------------------------

elif page == "Data Analysis":

    st.header("Electricity Consumption Data Analysis")
    st.write(
        "Explore the historical household electricity readings "
        "through summary statistics and time-based visualizations."
    )

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(
        df[["Global_active_power", "energy_kwh"]].describe(),
        use_container_width=True,
    )

    st.subheader("Daily Average Consumption")
    daily_average = (
        df["Global_active_power"]
        .resample("D")
        .mean()
        .dropna()
    )
    st.line_chart(daily_average, height=300)

    st.subheader("Monthly Average Power Consumption")
    monthly_average = (
        df["Global_active_power"]
        .resample("ME")
        .mean()
        .dropna()
    )
    st.bar_chart(monthly_average, height=300)

    st.subheader("Consumption by Hour of Day")
    hourly_profile = (
        df["Global_active_power"]
        .groupby(df.index.hour)
        .mean()
    )
    hourly_profile.index.name = "Hour of day"
    st.bar_chart(hourly_profile, height=300)

    st.subheader("Missing Values")
    missing_values = df.isna().sum().rename("Missing values")
    st.dataframe(missing_values.to_frame(), use_container_width=True)

# ========================= PART 3: PREDICTION, MODEL COMPARISON,
#                         ABOUT PROJECT =========================

# ---------------------------- ENERGY PREDICTION -------------------------------

elif page == "Energy Prediction":

    st.header("Predict Household Energy Consumption")
    st.write(
        "Choose manual prediction by entering previous readings, "
        "or automatically use the latest available historical readings."
    )

    prediction_mode = st.radio(
        "Prediction method",
        ["Automatic Prediction", "Manual Prediction"],
        horizontal=True,
    )

    st.divider()

    if prediction_mode == "Automatic Prediction":

        st.subheader("Automatic Energy Prediction")
        st.write(
            "The application uses the latest three available hourly "
            "readings from the historical dataset."
        )

        model_choice = st.selectbox(
            "Select ML Model",
            ["Linear Regression", "Random Forest"],
            key="automatic_model",
        )

        tariff = st.number_input(
            "Electricity tariff (currency per kWh)",
            min_value=0.0,
            value=8.0,
            step=0.5,
            key="automatic_tariff",
        )

        if st.button(
            "Generate Automatic Prediction",
            type="primary",
            key="generate_automatic",
        ):

            if len(hourly_data) < 3:
                st.error(
                    "Not enough valid hourly historical readings "
                    "to generate a prediction."
                )
            else:
                # Select the last three non-missing hourly values.
                recent_values = hourly_data.dropna().tail(3)

                if len(recent_values) < 3:
                    st.error(
                        "Three valid hourly readings are required."
                    )
                else:
                    lag_1 = float(recent_values.iloc[-1])
                    lag_2 = float(recent_values.iloc[-2])
                    lag_3 = float(recent_values.iloc[-3])

                    next_time = (
                        pd.Timestamp(recent_values.index[-1])
                        + pd.Timedelta(hours=1)
                    )

                    feature_row = make_feature_row(
                        next_time, lag_1, lag_2, lag_3
                    )

                    try:
                        prediction = predict_with_selected_model(
                            model_choice, feature_row
                        )

                        st.success(
                            "Automatic prediction generated successfully!"
                        )
                        st.subheader("Prediction Result")

                        result_col1, result_col2 = st.columns(2)
                        result_col1.metric(
                            "Prediction date and time",
                            next_time.strftime("%d-%m-%Y %H:%M"),
                        )
                        result_col2.metric(
                            "Predicted next-hour value",
                            f"{prediction:.3f}",
                        )

                        st.metric(
                            "Estimated electricity cost",
                            f"{prediction * tariff:.2f}",
                        )

                        st.caption(
                            "The prediction uses the latest three available "
                            "hourly readings and the selected saved ML model."
                        )

                        with st.expander("View generated input features"):
                            st.dataframe(
                                feature_row,
                                use_container_width=True,
                            )

                    except Exception as error:
                        st.error(
                            f"Unable to generate prediction: {error}"
                        )

    else:

        st.subheader("Manual Energy Prediction")
        st.write(
            "Enter the previous hourly consumption readings "
            "to estimate the next-hour value."
        )

        left_col, right_col = st.columns(2)

        with left_col:
            prediction_date = st.date_input(
                "Prediction Date",
                key="manual_prediction_date",
            )

            prediction_hour = st.slider(
                "Prediction Hour",
                min_value=0,
                max_value=23,
                value=12,
                key="manual_prediction_hour",
            )

            lag_1 = st.number_input(
                "Previous Hour Consumption",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key="manual_lag_1",
            )

            lag_2 = st.number_input(
                "Consumption 2 Hours Ago",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key="manual_lag_2",
            )

        with right_col:
            lag_3 = st.number_input(
                "Consumption 3 Hours Ago",
                min_value=0.0,
                value=1.0,
                step=0.1,
                key="manual_lag_3",
            )

            model_choice = st.selectbox(
                "Select ML Model",
                ["Linear Regression", "Random Forest"],
                key="manual_model",
            )

            tariff = st.number_input(
                "Electricity tariff (currency per kWh)",
                min_value=0.0,
                value=8.0,
                step=0.5,
                key="manual_tariff",
            )

        if st.button(
            "Predict Energy Consumption",
            type="primary",
            key="generate_manual",
        ):

            prediction_timestamp = pd.Timestamp(
                year=prediction_date.year,
                month=prediction_date.month,
                day=prediction_date.day,
                hour=prediction_hour,
            )

            feature_row = make_feature_row(
                prediction_timestamp,
                lag_1,
                lag_2,
                lag_3,
            )

            try:
                prediction = predict_with_selected_model(
                    model_choice, feature_row
                )

                st.success(
                    f"Predicted next-hour consumption: {prediction:.3f}"
                )
                st.metric(
                    "Estimated electricity cost",
                    f"{prediction * tariff:.2f}",
                )

                with st.expander("View generated input features"):
                    st.dataframe(
                        feature_row,
                        use_container_width=True,
                    )

            except Exception as error:
                st.error(f"Unable to generate prediction: {error}")


# ---------------------------- MODEL COMPARISON -------------------------------

elif page == "Model Comparison":

    st.header("Machine Learning Model Comparison")
    st.write(
        "Compare Linear Regression and Random Forest using "
        "the evaluation metrics generated during model training."
    )

    if MODEL_RESULTS_PATH.exists():

        try:
            results = pd.read_csv(MODEL_RESULTS_PATH)

            required_metrics = {
                "Model", "MAE", "MSE", "RMSE", "R2 Score"
            }

            if required_metrics.issubset(results.columns):

                st.subheader("Model Performance Metrics")
                st.dataframe(
                    results.round(4),
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader("R² Score Comparison")
                r2_chart = results.set_index("Model")[["R2 Score"]]
                st.bar_chart(r2_chart, height=300)

                st.subheader("Error Metric Comparison")
                error_columns = ["MAE", "MSE", "RMSE"]
                error_chart = results.set_index("Model")[error_columns]
                st.bar_chart(error_chart, height=300)

                best_row = results.loc[results["RMSE"].idxmin()]
                st.success(
                    f"Based on RMSE, {best_row['Model']} performs better "
                    f"among the evaluated models "
                    f"(RMSE: {best_row['RMSE']:.4f})."
                )

            else:
                st.warning(
                    "model_comparison.csv exists but does not contain "
                    "the expected columns: Model, MAE, MSE, RMSE, R2 Score."
                )

        except Exception as error:
            st.error(f"Could not read model comparison results: {error}")

    else:
        st.warning(
            "Model comparison results were not found. "
            "Run train_model.py locally and upload "
            "results/model_comparison.csv to GitHub."
        )


# ---------------------------- ABOUT PROJECT ----------------------------------

elif page == "About Project":

    st.header("About the Project")

    st.subheader("Project Title")
    st.write(
        "Smart Energy Consumption Analysis and Prediction Using "
        "Machine Learning"
    )

    st.subheader("Objective")
    st.write(
        "The objective of this project is to analyze historical household "
        "electricity consumption patterns and predict future consumption "
        "using machine learning algorithms."
    )

    st.subheader("Dataset")
    st.write(
        "The project uses the Individual Household Electric Power "
        "Consumption dataset from the UCI Machine Learning Repository."
    )

    st.subheader("Machine Learning Algorithms")
    st.write(
        "1. Linear Regression\n\n"
        "2. Random Forest Regression"
    )

    st.subheader("Technologies Used")
    st.write(
        "- Python\n"
        "- Pandas and NumPy\n"
        "- Scikit-learn\n"
        "- Matplotlib\n"
        "- Streamlit\n"
        "- Joblib"
    )

    st.subheader("Evaluation Metrics")
    st.write(
        "- Mean Absolute Error (MAE)\n"
        "- Mean Squared Error (MSE)\n"
        "- Root Mean Squared Error (RMSE)\n"
        "- R-squared (R²) Score"
    )

    st.subheader("Project Outcome")
    st.write(
        "The application provides historical consumption analysis, "
        "visualizations, comparison of trained machine learning models, "
        "and manual and automatic next-hour predictions."
    )

    st.caption(
        "Academic mini project — Smart Energy Consumption Analysis "
        "and Prediction Using Machine Learning."
    )