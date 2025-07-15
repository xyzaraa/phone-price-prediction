# Phone Price Prediction

This project aims to predict mobile phone prices using various specifications and features of mobile devices. The analysis involves data cleaning, exploratory data analysis, feature engineering, and the implementation of machine learning models for price prediction.

## Table of Contents

- [Data Source](#data-source)
- [Project Overview](#project-overview)
- [Key Steps and Analysis](#key-steps-and-analysis)
  - [Data Loading and Initial Exploration](#data-loading-and-initial-exploration)
  - [Data Cleaning](#data-cleaning)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Feature Engineering](#feature-engineering)
  - [Feature Selection](#feature-selection)
  - [Modelling](#modelling)
  - [Model Evaluation](#model-evaluation)
- [Results and Insights](#results-and-insights)
- [Technologies and Libraries Used](#technologies-and-libraries-used)

## Data Source

The dataset used in this project is `mobile.csv`, sourced from `/kaggle/input/mobile-uncleaned-data-set-scrapped-real-website/mobile.csv`.

## Project Overview

The main objective of this project is to build and evaluate predictive models that can estimate the price of mobile phones based on their technical specifications and other attributes.

## Key Steps and Analysis

### Data Loading and Initial Exploration

The project begins by loading the `mobile.csv` dataset into a pandas DataFrame. Initial exploration involves viewing the first few rows of the data (`df.head()`) to understand its structure and identifying missing values across different columns (`df.isnull().sum()`). Basic descriptive statistics for numerical columns like 'Spec Score', 'rating', and 'price' are also generated (`df.describe()`).

### Data Cleaning

- Outlier Handling: Outliers in the 'Spec Score' column were identified using the IQR (Interquartile Range) method. Values below the lower bound (60.0) were capped at the lower bound.
- Missing Values Imputation:
  - The 'fm' (FM Radio) column, which had a significant number of missing values, was dropped due to high sparsity.
  - Missing values in 'storage', 'processor', 'memoryExternal', 'battery', 'display', 'camera', and 'version' columns were imputed with 'Unknown' or appropriate strategies (e.g., median for numerical features extracted later).

### Exploratory Data Analysis (EDA)

- Price and Spec Score Correlation: A scatter plot of 'Log10 Price' vs. 'Spec Score' revealed a positive and generally linear relationship, indicating that higher spec scores are associated with higher prices. Logarithmic transformation was applied to price to reduce data accumulation at lower values and improve clarity.
- Processor and Tag Distribution: Bar plots were generated to visualize the distribution of the top 10 processor types and the count of different tags (e.g., 'UPCOMING', 'LAUNCHED').

### Feature Engineering

New numerical features were extracted from existing text-based columns:
- `Battery_mAh`: Extracted from the 'battery' column.
- `Display_Inches`: Extracted from the 'display' column.
- `Camera_MP`: Extracted from the 'camera' column.
- `RAM_GB` and `Internal_Storage_GB`: Extracted from the 'storage' column.

Categorical features for `Processor_Brand` (e.g., 'Snapdragon', 'Dimensity') and `Version_Main` (e.g., 'Android 15', 'iOS 17') were also engineered from 'processor' and 'version' columns, respectively.

Missing values in these newly engineered numerical features were imputed using the median, and categorical features were filled with 'Unknown'.

### Feature Selection

The following features were selected for the prediction model:
- Numerical Features: 'Spec Score', 'rating', 'Battery_mAh', 'Display_Inches', 'Camera_MP', 'RAM_GB', 'Internal_Storage_GB'.
- Categorical Features: 'tag', 'sim', 'memoryExternal', 'Processor_Brand', 'Version_Main'.
- Target Variable: 'price'.

The data was split into training (80%) and testing (20%) sets.

### Modelling

Two machine learning models were used for price prediction:
1. Linear Regression
2. XGBoost Regressor

Preprocessing pipelines were established for both numerical (imputation with median, scaling with StandardScaler) and categorical (imputation with most frequent, one-hot encoding with OneHotEncoder) features. These pipelines were integrated with the regressors.

### Model Evaluation

Both models were trained and evaluated using the following metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R2) Score

Scatter plots comparing actual prices against predicted prices were also generated for visual assessment.

## Results and Insights

The model evaluation results are as follows:

| Model                | MAE            | MSE               | RMSE            | R2 Score        |
| :------------------- | :------------- | :---------------- | :-------------- | :-------------- |
| Linear Regression    | 17735.78       | 862568812.14      | 29369.52        | 0.5387          |
| XGBoost Regressor    | 11737.64       | 1117538069.47     | 33429.60        | 0.4023          |

Based on the R2 score, the Linear Regression model performed slightly better (R2: 0.5387) compared to the XGBoost Regressor (R2: 0.4023) in explaining the variance in phone prices. This indicates that a simpler linear relationship might be more prevalent in the dataset for price prediction given the current features and model parameters. The scatter plots also visually confirm the model's performance by showing the spread of predicted values against actual values.

## Technologies and Libraries Used

- Python
- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn (for model selection, preprocessing, and metrics)
- XGBoost

---

Note: The results and insights are based on the execution of the provided notebook. Further hyperparameter tuning and feature engineering could potentially improve model performance.
