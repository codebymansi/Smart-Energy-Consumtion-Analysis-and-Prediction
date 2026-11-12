# Smart Energy Consumption Analysis and Prediction Using Machine Learning

## Project Overview

This project focuses on analyzing household electricity consumption patterns and predicting future energy usage using Machine Learning algorithms.

The system uses historical electricity consumption data to identify usage patterns, analyze energy trends, and predict consumption.

## Objectives

- Analyze household electricity consumption patterns.
- Perform data cleaning and preprocessing.
- Visualize electricity consumption trends.
- Predict energy consumption using Machine Learning.
- Compare the performance of different ML algorithms.
- Estimate electricity costs based on predicted consumption.

## Dataset

**Individual Household Electric Power Consumption Dataset**

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

Download and extract the dataset. Place `household_power_consumption.txt` inside the `dataset` folder.

## Machine Learning Algorithms

1. Linear Regression
2. Random Forest Regression

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## Model Evaluation

The models are evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

### Model Comparison

| Model | MAE | MSE | RMSE | R² Score |
|---|---:|---:|---:|---:|
| Linear Regression | 0.3980 | 0.3009 | 0.5485 | 0.4562 |
| Random Forest | 0.3395 | 0.2380 | 0.4878 | 0.5698 |

Random Forest achieved better results across all four evaluation metrics in this experiment.

## Project Structure

```text
Smart-Energy-Consumption/
├── dataset/
├── models/
├── results/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/codebymansi/Smart-Energy-Consumtion-Analysis-and-Prediction.git
cd Smart-Energy-Consumtion-Analysis-and-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Download the UCI dataset and place the extracted file at:

`dataset/household_power_consumption.txt`

### 4. Train the models

```bash
python train_model.py
```

### 5. Launch the dashboard

```bash
streamlit run app.py
```

## Project Outputs

- Household electricity consumption analysis.
- Energy consumption visualizations.
- Machine Learning predictions.
- Model performance comparison.
- Electricity cost estimation.

## Author

Developed as a college mini-project.

