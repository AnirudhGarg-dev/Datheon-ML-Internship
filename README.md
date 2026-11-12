# Datheon-ML-Internship
A collection of practical machine learning projects completed as part of the Datheon AI/ML Internship, covering regression, classification, feature engineering, model evaluation, and NLP.

# California Housing Price Prediction

## Datheon ML Internship — Task 2

A machine learning regression project that predicts California median house values using the California Housing dataset. The project compares Linear Regression, Ridge Regression, and Lasso Regression models and evaluates their performance using MAE, RMSE, and R².

## Project Objective

The objective of this project is to build an end-to-end housing price prediction pipeline involving:

- Exploratory Data Analysis
- Data preprocessing
- Missing-value handling
- Categorical encoding
- Feature scaling
- Feature engineering
- Linear Regression
- Ridge Regression
- Lasso Regression
- Hyperparameter tuning using GridSearchCV
- Residual analysis
- Model coefficient analysis

## Dataset

The project uses the California Housing dataset containing 20,640 observations and 10 original columns.

### Target Variable

`median_house_value`

### Original Features

- longitude
- latitude
- housing_median_age
- total_rooms
- total_bedrooms
- population
- households
- median_income
- ocean_proximity

The dataset contains missing values in `total_bedrooms`, which are handled using median imputation.

## Feature Engineering

Three additional features were created:

- `rooms_per_household`
- `bedrooms_per_room`
- `population_per_household`

These ratio-based features provide additional information about housing composition and household characteristics.

## Preprocessing

A Scikit-Learn preprocessing pipeline was implemented.

### Numerical Features

- Median imputation
- StandardScaler

### Categorical Features

- Most-frequent imputation
- OneHotEncoder

The preprocessing pipeline is integrated with the regression models to prevent data leakage.

## Models

Three regression algorithms were evaluated:

### 1. Linear Regression

Used as the baseline regression model.

### 2. Ridge Regression

Uses L2 regularization to reduce the effect of large model coefficients.

### 3. Lasso Regression

Uses L1 regularization and can perform implicit feature selection by shrinking coefficients toward zero.

GridSearchCV was used to tune the regularization parameter (`alpha`) for Ridge and Lasso regression.

## Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 49,645.49 | 69,127.04 | 0.6353 |
| Lasso Regression | 49,646.43 | 69,128.09 | 0.6353 |
| Lasso Tuned | 49,646.43 | 69,128.09 | 0.6353 |
| Ridge Regression | 49,652.14 | 69,134.97 | 0.6353 |
| Ridge Tuned | 49,652.14 | 69,134.97 | 0.6353 |

### Best Model

Linear Regression achieved the lowest test RMSE and MAE among the evaluated models.

**RMSE:** 69,127.04

**MAE:** 49,645.49

**R²:** 0.6353

The model explains approximately 63.5% of the variance in median house values on the test set.

## Residual Analysis

The selected Linear Regression model produced the following residual statistics:

- Mean residual: 783.76
- Residual standard deviation: 69,130.97
- Maximum residual: 453,363.64
- Minimum residual: -414,151.57

The mean residual is close to zero, suggesting limited overall prediction bias. However, the large residual range indicates that the model has difficulty accurately predicting some individual house values.

## Visualizations

The project includes:

- Geographical distribution of California housing prices
- Correlation heatmap
- Median house value distribution
- Actual vs predicted values
- Residual distribution
- Residuals vs predicted values
- Model performance comparison
- Linear regression feature coefficients

## Key Findings

- Median income is one of the strongest numerical predictors of median house value.
- Geographic variables such as latitude and longitude contribute substantially to prediction.
- Ocean proximity categories provide meaningful information.
- Feature engineering adds useful household-level ratios.
- Ridge and Lasso regularization produced performance very similar to baseline Linear Regression.
- The final Linear Regression model achieved an R² of approximately 0.635.

## Limitations

The model does not capture all nonlinear relationships between geographic, demographic, and housing characteristics.

The dataset also contains a ceiling on the target variable at approximately $500,001, which can limit prediction accuracy for higher-value properties.

## Project Structure

```text
Datheon-ML-Internship/
│
├── notebooks/
│   └── Task_2_Housing_Price_Prediction.ipynb
│
├── models/
│   └── california_housing_linear_regression.joblib
│
├── reports/
│   └── figures/
│       ├── geographical_distribution.png
│       ├── correlation_heatmap.png
│       ├── actual_vs_predicted.png
│       ├── residual_distribution.png
│       ├── residuals_vs_predicted.png
│       ├── model_comparison.png
│       └── feature_coefficients.png
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Open the Jupyter notebook:

```bash
jupyter notebook
```

Then open:

```text
notebooks/Task_2_Housing_Price_Prediction.ipynb
```

Run the notebook cells sequentially to reproduce the analysis and model training.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- SciPy
- Joblib
- Jupyter Notebook

## Internship Deliverable

This project was completed as part of the Datheon ML Internship Task Pack.

The repository demonstrates an end-to-end regression workflow from exploratory data analysis through model evaluation and model persistence.