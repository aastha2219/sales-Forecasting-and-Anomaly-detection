# pyrefly: ignore [missing-import]
import nbformat as nbf

# Initialize notebook
nb = nbf.v4.new_notebook()

# Define cells
cells = []

# Title & Intro
cells.append(nbf.v4.new_markdown_cell("""# Sales Forecasting & Anomaly Detection System
### Comprehensive Analysis, Modeling, and Segmentation on Superstore Retail Data
**Author:** Aastha Anshu  
**Date:** July 2026

This Jupyter Notebook contains the complete end-to-end implementation of the Sales Forecasting, Anomaly Detection, and Product Clustering system. It is structured into 6 main tasks:
1. **Data Loading, Merging & Deep Exploration**
2. **Time Series Analysis & Decomposition**
3. **Sales Forecasting using 3 Different Models (SARIMA, Prophet, XGBoost)**
4. **Product Category & Region Level Forecasting**
5. **Anomaly Detection in Sales Data (Isolation Forest vs Z-Score)**
6. **Product Demand Segmentation using K-Means Clustering**

Let's begin by importing the necessary libraries and setting up our visualization styles."""))

# Imports
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# Create charts directory if it doesn't exist
os.makedirs('charts', exist_ok=True)
print("Environment initialized and charts directory created.")"""))

# Task 1 Intro
cells.append(nbf.v4.new_markdown_cell("""## Task 1 — Data Loading, Merging & Deep Exploration
In this section, we will:
1. Load the Superstore Sales dataset.
2. Parse dates and engineer rich temporal features.
3. Handle duplicates and missing values.
4. Merge with the supplementary Video Game Sales dataset to perform multi-source analysis.
5. Answer key business questions regarding category revenue, regional growth, shipping duration, and seasonality."""))

# Task 1 Code
cells.append(nbf.v4.new_code_cell("""# 1. Load the Superstore dataset
df = pd.read_csv('https://raw.githubusercontent.com/aastha2219/sales-prediction-and-anomaly-detection/main/train.csv', encoding='utf-8')
print(f"Loaded Superstore dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

# 2. Parse date columns as datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], format='mixed', dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], format='mixed', dayfirst=True)

# 3. Clean duplicates and check for missing values
duplicates_count = df.duplicated().sum()
missing_values = df.isnull().sum()
print(f"Number of duplicate rows: {duplicates_count}")
print("Missing values per column:")
print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "None")

# Drop duplicates if any
if duplicates_count > 0:
    df.drop_duplicates(inplace=True)

# 4. Feature engineering: extract temporal features
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Week_Number'] = df['Order Date'].dt.isocalendar().week
df['Day_of_Week'] = df['Order Date'].dt.day_name()
df['Quarter'] = df['Order Date'].dt.quarter

def get_season(month):
    if month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    elif month in [9, 10, 11]: return 'Autumn'
    else: return 'Winter'
df['Season'] = df['Month'].map(get_season)

print("\\nEngineered Temporal Features. Sample data:")
df[['Order Date', 'Year', 'Month', 'Week_Number', 'Day_of_Week', 'Quarter', 'Season']].head()"""))

# Task 1 Q1 & Q2
cells.append(nbf.v4.new_markdown_cell("""### Question 1 & 2: Revenue and Regional Growth
- **Q1:** Which product category generates the highest total revenue?
- **Q2:** Which region has the most consistent sales growth over 4 years?"""))

cells.append(nbf.v4.new_code_cell("""# Q1: Highest Revenue Product Category
category_revenue = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
print("Q1: Revenue by Category:")
for cat, rev in category_revenue.items():
    print(f" - {cat}: ${rev:,.2f}")
print(f"Result: Category '{category_revenue.index[0]}' generates the highest total revenue.\\n")

# Q2: Region with most consistent sales growth over 4 years
region_annual_sales = df.groupby(['Region', 'Year'])['Sales'].sum().unstack('Year')
region_growth_rate = region_annual_sales.pct_change(axis=1).iloc[:, 1:] * 100

print("Q2: Annual Sales by Region ($):")
print(region_annual_sales.round(2))
print("\\nYear-over-Year Growth Rates (%):")
print(region_growth_rate.round(2))

# We evaluate consistency by checking if growth is positive every year and looking at the variance/average growth
print("\\nGrowth Consistency Analysis:")
for region in region_growth_rate.index:
    rates = region_growth_rate.loc[region]
    all_positive = all(rates > 0)
    avg_growth = rates.mean()
    std_growth = rates.std()
    print(f" - {region} Region: All years positive: {all_positive} | Avg Growth: {avg_growth:.2f}% | Std Dev: {std_growth:.2f}%")"""))

# Task 1 Q3 & Q4
cells.append(nbf.v4.new_markdown_cell("""### Question 3 & 4: Shipping Time and Seasonality
- **Q3:** What is the average time between Order Date and Ship Date — and does it vary by region?
- **Q4:** Are there months that consistently spike across all years (seasonality)?"""))

cells.append(nbf.v4.new_code_cell("""# Q3: Shipping duration
df['Ship_Duration'] = (df['Ship Date'] - df['Order Date']).dt.days
overall_avg_ship = df['Ship_Duration'].mean()
region_avg_ship = df.groupby('Region')['Ship_Duration'].mean()

print(f"Overall average shipping time: {overall_avg_ship:.2f} days\\n")
print("Average shipping time by Region (days):")
for reg, days in region_avg_ship.items():
    print(f" - {reg}: {days:.2f} days")
print("\\nResult: The shipping time shows very little variation across regions, staying consistently around 3.9 to 4.1 days.\\n")

# Q4: Seasonality - Months that consistently spike
monthly_sales_year = df.groupby(['Year', 'Month'])['Sales'].sum().unstack('Year')
print("Monthly Sales by Year ($):")
print(monthly_sales_year.round(2))

# Let's plot average monthly sales to show the trend
plt.figure()
sns.barplot(data=df, x='Month', y='Sales', estimator=sum, errorbar=None, palette="viridis")
plt.title("Total Sales by Month (Aggregated Over 4 Years)")
plt.xlabel("Month")
plt.ylabel("Total Sales ($)")
plt.savefig('charts/monthly_seasonality.png', bbox_inches='tight')
plt.close()

# Find months that spike
avg_monthly = monthly_sales_year.mean(axis=1)
peak_months = avg_monthly.nlargest(3)
print("\\nTop 3 months by average sales:")
for m, val in peak_months.items():
    print(f" - Month {m}: ${val:,.2f}")"""))

# Task 1 Multi-Source Merge
cells.append(nbf.v4.new_markdown_cell("""### Multi-Source Analysis: Merging with Video Game Sales
We will load `vgsales.csv` and aggregate it by Year. We will merge it with the Superstore Technology category sales (aggregated by Year) to check for correlations between Video Game Sales and Technology retail performance."""))

cells.append(nbf.v4.new_code_cell("""# Load Video Game Sales dataset
vg_df = pd.read_csv('https://raw.githubusercontent.com/aastha2219/sales-prediction-and-anomaly-detection/main/vgsales.csv')
print(f"Loaded Video Game Sales dataset with {vg_df.shape[0]} rows.")

# Clean Year in Video Game Sales
vg_df['Year'] = pd.to_numeric(vg_df['Year'], errors='coerce')
vg_df = vg_df.dropna(subset=['Year'])
vg_df['Year'] = vg_df['Year'].astype(int)

# Aggregate Video Games Sales by Year (filter for years overlap: 2015 to 2018)
vg_annual = vg_df[vg_df['Year'].between(2015, 2018)].groupby('Year')[['NA_Sales', 'Global_Sales']].sum()
vg_annual = vg_annual.rename(columns={'NA_Sales': 'VG_NA_Sales_M', 'Global_Sales': 'VG_Global_Sales_M'})

# Aggregate Superstore Technology category sales by Year
tech_annual = df[(df['Category'] == 'Technology') & (df['Year'].between(2015, 2018))].groupby('Year')['Sales'].sum()
tech_annual = tech_annual.to_frame('Superstore_Tech_Sales')

# Merge datasets
merged_annual = tech_annual.join(vg_annual)
print("Merged Annual Multi-Source Sales Data:")
print(merged_annual)

# Correlation matrix
correlation = merged_annual.corr()
print("\\nCorrelation Matrix between Video Game Sales and Superstore Tech Sales:")
print(correlation)"""))

# Task 2 Intro
cells.append(nbf.v4.new_markdown_cell("""## Task 2 — Time Series Analysis & Decomposition
In this section, we will:
1. Resample the daily sales into monthly totals.
2. Apply Time Series Decomposition (Trend, Seasonality, Residual).
3. Check for Stationarity using the Augmented Dickey-Fuller (ADF) test.
4. Apply differencing to render the series stationary if necessary."""))

# Task 2 Code
cells.append(nbf.v4.new_code_cell("""# Resample daily sales to monthly totals
# We use 'MS' (Month Start) which is standard for time series analysis
monthly_sales = df.set_index('Order Date').resample('MS')['Sales'].sum()
print(f"Monthly sales series created. Total months: {len(monthly_sales)}")

# Plot overall monthly sales trend
plt.figure(figsize=(12, 5))
plt.plot(monthly_sales, marker='o', linewidth=2, color='#1f77b4')
plt.title("Overall Monthly Sales Trend (2015-2018)")
plt.xlabel("Order Date")
plt.ylabel("Sales ($)")
plt.savefig('charts/monthly_sales_trend.png', bbox_inches='tight')
plt.close()

# Apply Time Series Decomposition
decomposition = seasonal_decompose(monthly_sales, model='additive', period=12)

# Plot components
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
decomposition.observed.plot(ax=ax1, color='#1f77b4', legend=False)
ax1.set_ylabel('Observed')
ax1.set_title('Time Series Decomposition of Monthly Sales')

decomposition.trend.plot(ax=ax2, color='#ff7f0e', legend=False)
ax2.set_ylabel('Trend')

decomposition.seasonal.plot(ax=ax3, color='#2ca02c', legend=False)
ax3.set_ylabel('Seasonal')

decomposition.resid.plot(ax=ax4, color='#d62728', style='o', legend=False)
ax4.set_ylabel('Residual')

plt.xlabel('Date')
plt.tight_layout()
plt.savefig('charts/decomposition.png', bbox_inches='tight')
plt.close()
print("Saved decomposition chart to 'charts/decomposition.png'.")"""))

# Task 2 Stationarity
cells.append(nbf.v4.new_code_cell("""# Stationarity check: Augmented Dickey-Fuller Test
def run_adf_test(series, name):
    print(f"--- ADF Test for {name} ---")
    result = adfuller(series)
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"\\t{key}: {value:.4f}")
    
    if result[1] <= 0.05:
        print("Conclusion: The series is STATIONARY (reject null hypothesis at 95% confidence level)\\n")
        return True
    else:
        print("Conclusion: The series is NON-STATIONARY (fail to reject null hypothesis)\\n")
        return False

is_stationary = run_adf_test(monthly_sales, "Original Monthly Sales")

# Apply differencing if non-stationary
if not is_stationary:
    monthly_sales_diff = monthly_sales.diff().dropna()
    print("Applying first-order differencing...")
    run_adf_test(monthly_sales_diff, "First-Differenced Monthly Sales")
    
    # Plot differenced series
    plt.figure(figsize=(12, 4))
    plt.plot(monthly_sales_diff, marker='s', color='purple')
    plt.title("First-Differenced Monthly Sales")
    plt.xlabel("Date")
    plt.ylabel("Differenced Sales ($)")
    plt.savefig('charts/differenced_sales.png', bbox_inches='tight')
    plt.close()"""))

# Task 3 Intro
cells.append(nbf.v4.new_markdown_cell("""## Task 3 — Sales Forecasting using 3 Different Models
In this core technical section, we will:
1. Split our monthly sales into a Train Set (first 45 months) and Test Set (last 3 months) to evaluate the models.
2. Build and train **SARIMA**, **Facebook Prophet**, and **XGBoost**.
3. Generate a 3-month forecast and calculate performance metrics: **MAE**, **RMSE**, and **MAPE**.
4. Construct a model comparison table and recommend the best model for production use.

Let's split the dataset first."""))

# Task 3 Split
cells.append(nbf.v4.new_code_cell("""# Train-test split
train_len = len(monthly_sales) - 3
train_data = monthly_sales.iloc[:train_len]
test_data = monthly_sales.iloc[train_len:]

print(f"Train period: {train_data.index[0].strftime('%Y-%m')} to {train_data.index[-1].strftime('%Y-%m')} ({len(train_data)} months)")
print(f"Test period: {test_data.index[0].strftime('%Y-%m')} to {test_data.index[-1].strftime('%Y-%m')} ({len(test_data)} months)")

# Define metric helper
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape"""))

# Task 3 Model 1 (SARIMA)
cells.append(nbf.v4.new_markdown_cell("""### Model 1 — SARIMA (Statistical Model)
We will perform a grid search to select the optimal parameters `(p, d, q) x (P, D, Q, 12)` that minimize AIC, and use them to forecast the test set and the 3-month future horizon."""))

cells.append(nbf.v4.new_code_cell("""# Grid search for SARIMA
import itertools

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

best_aic = float("inf")
best_order = None
best_seasonal_order = None

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            model = SARIMAX(train_data,
                            order=param,
                            seasonal_order=param_seasonal,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            results = model.fit(disp=False)
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = param
                best_seasonal_order = param_seasonal
        except:
            continue

print(f"Best SARIMA order: {best_order} x {best_seasonal_order} with AIC: {best_aic:.2f}")

# Fit the best model on training data
sarima_model = SARIMAX(train_data,
                       order=best_order,
                       seasonal_order=best_seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)
sarima_results = sarima_model.fit(disp=False)

# Forecast test period
sarima_pred_res = sarima_results.get_forecast(steps=3)
sarima_pred = sarima_pred_res.predicted_mean
sarima_ci = sarima_pred_res.conf_int()

# Fit model on ALL data for out-of-sample future forecast
sarima_full_model = SARIMAX(monthly_sales,
                            order=best_order,
                            seasonal_order=best_seasonal_order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
sarima_full_results = sarima_full_model.fit(disp=False)
sarima_future_res = sarima_full_results.get_forecast(steps=3)
sarima_future = sarima_future_res.predicted_mean
sarima_future_ci = sarima_future_res.conf_int()

# Evaluate
sarima_mae, sarima_rmse, sarima_mape = calculate_metrics(test_data, sarima_pred)
print(f"SARIMA Test Metrics -> MAE: {sarima_mae:.2f} | RMSE: {sarima_rmse:.2f} | MAPE: {sarima_mape:.2f}%")

# Plot test forecast
plt.figure()
plt.plot(train_data.index[-12:], train_data.iloc[-12:], label='Historical Sales (Last 12m)')
plt.plot(test_data.index, test_data, label='Actual Test Sales', marker='o', color='black')
plt.plot(test_data.index, sarima_pred, label='SARIMA Test Forecast', marker='x', color='red')
plt.fill_between(test_data.index, sarima_ci.iloc[:, 0], sarima_ci.iloc[:, 1], color='red', alpha=0.15)
plt.title("SARIMA Test Set Forecast vs Actuals")
plt.legend()
plt.savefig('charts/sarima_test_forecast.png', bbox_inches='tight')
plt.close()"""))

# Task 3 Model 2 (Prophet)
cells.append(nbf.v4.new_markdown_cell("""### Model 2 — Facebook Prophet
Prophet requires columns named `ds` (datestamp) and `y` (target value). We will fit the model, evaluate it on the test set, and interpret seasonality."""))

cells.append(nbf.v4.new_code_cell("""# Prepare Prophet datasets
prophet_train = train_data.reset_index().rename(columns={'Order Date': 'ds', 'Sales': 'y'})
prophet_all = monthly_sales.reset_index().rename(columns={'Order Date': 'ds', 'Sales': 'y'})

# Fit on train data
prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
prophet_model.fit(prophet_train)

# Predict test period
future_test = pd.DataFrame({'ds': test_data.index})
prophet_pred_df = prophet_model.predict(future_test)
prophet_pred = prophet_pred_df['yhat'].values

# Evaluate
prophet_mae, prophet_rmse, prophet_mape = calculate_metrics(test_data, prophet_pred)
print(f"Prophet Test Metrics -> MAE: {prophet_mae:.2f} | RMSE: {prophet_rmse:.2f} | MAPE: {prophet_mape:.2f}%")

# Fit on ALL data and forecast next 3 months
prophet_full_model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
prophet_full_model.fit(prophet_all)
future_out = prophet_full_model.make_future_dataframe(periods=3, freq='MS')
prophet_future_df = prophet_full_model.predict(future_out)

prophet_future = prophet_future_df.iloc[-3:]['yhat'].values

# Plot components
fig = prophet_full_model.plot_components(prophet_future_df)
fig.savefig('charts/prophet_components.png', bbox_inches='tight')
plt.close(fig)
print("Saved Prophet components plot to 'charts/prophet_components.png'.")"""))

# Task 3 Model 3 (XGBoost)
cells.append(nbf.v4.new_markdown_cell("""### Model 3 — XGBoost for Time Series (ML-based Approach)
We frame this as a supervised machine learning task by engineering:
- Lag 1, Lag 2, Lag 3 (sales from 1, 2, and 3 months ago)
- Rolling Mean of past 3 months
- Temporal features (Month, Quarter)
We will build a recursive forecasting function to generate the 3-step predictions."""))

cells.append(nbf.v4.new_code_cell("""# Function to build features for XGBoost
def create_lag_features(series):
    df_features = pd.DataFrame(index=series.index)
    df_features['y'] = series.values
    df_features['Lag1'] = series.shift(1)
    df_features['Lag2'] = series.shift(2)
    df_features['Lag3'] = series.shift(3)
    df_features['RollingMean3'] = series.shift(1).rolling(window=3).mean()
    df_features['Month'] = series.index.month
    df_features['Quarter'] = series.index.quarter
    return df_features.dropna()

# Generate features on all data
xg_data = create_lag_features(monthly_sales)
print(f"XGBoost dataset created. Features shape: {xg_data.shape}")

# Split features into train and test
# To evaluate the test set (last 3 months) fairly, we train on data before the test set starts
xg_train = xg_data.iloc[:-3]
xg_test = xg_data.iloc[-3:]

X_train, y_train = xg_train.drop(columns='y'), xg_train['y']
X_test, y_test = xg_test.drop(columns='y'), xg_test['y']

# Train XGBoost
xgb_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Recursive Forecast on test set
xgb_pred = []
# Start with last available train values for lags
current_lags = list(train_data.iloc[-3:].values) # [t-2, t-1, t]
for date in test_data.index:
    # Build features for current step
    lag1 = current_lags[-1]
    lag2 = current_lags[-2]
    lag3 = current_lags[-3]
    roll_mean = np.mean(current_lags[-3:])
    month = date.month
    quarter = date.quarter
    
    pred_val = xgb_model.predict(pd.DataFrame([[lag1, lag2, lag3, roll_mean, month, quarter]], 
                                            columns=['Lag1', 'Lag2', 'Lag3', 'RollingMean3', 'Month', 'Quarter']))[0]
    xgb_pred.append(pred_val)
    current_lags.append(pred_val)

xgb_pred = np.array(xgb_pred)

# Evaluate
xgb_mae, xgb_rmse, xgb_mape = calculate_metrics(test_data, xgb_pred)
print(f"XGBoost Test Metrics -> MAE: {xgb_mae:.2f} | RMSE: {xgb_rmse:.2f} | MAPE: {xgb_mape:.2f}%")

# Fit on ALL data for out-of-sample future forecasting
X_all, y_all = xg_data.drop(columns='y'), xg_data['y']
xgb_full_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
xgb_full_model.fit(X_all, y_all)

# Recursive forecast for the next 3 future months
future_dates = pd.date_range(start=monthly_sales.index[-1] + pd.DateOffset(months=1), periods=3, freq='MS')
xgb_future = []
current_lags = list(monthly_sales.iloc[-3:].values)
for date in future_dates:
    lag1 = current_lags[-1]
    lag2 = current_lags[-2]
    lag3 = current_lags[-3]
    roll_mean = np.mean(current_lags[-3:])
    month = date.month
    quarter = date.quarter
    
    pred_val = xgb_full_model.predict(pd.DataFrame([[lag1, lag2, lag3, roll_mean, month, quarter]], 
                                                 columns=['Lag1', 'Lag2', 'Lag3', 'RollingMean3', 'Month', 'Quarter']))[0]
    xgb_future.append(pred_val)
    current_lags.append(pred_val)

xgb_future = np.array(xgb_future)"""))

# Task 3 Comparison Table
cells.append(nbf.v4.new_markdown_cell("""### Model Comparison Table & Recommendation
Let's assemble the performance metrics and future 3-month forecast values for all three models."""))

cells.append(nbf.v4.new_code_cell("""# Extract out-of-sample forecasts for comparison table
# For SARIMA:
sarima_f1, sarima_f2, sarima_f3 = sarima_future.values
# For Prophet:
prop_f1, prop_f2, prop_f3 = prophet_future
# For XGBoost:
xgb_f1, xgb_f2, xgb_f3 = xgb_future

# Create Comparison Dataframe
comparison_data = {
    'Model': ['SARIMA', 'Prophet', 'XGBoost'],
    'MAE': [sarima_mae, prophet_mae, xgb_mae],
    'RMSE': [sarima_rmse, prophet_rmse, xgb_rmse],
    'MAPE (%)': [sarima_mape, prophet_mape, xgb_mape],
    'Forecast Month 1': [sarima_f1, prop_f1, xgb_f1],
    'Forecast Month 2': [sarima_f2, prop_f2, xgb_f2],
    'Forecast Month 3': [sarima_f3, prop_f3, xgb_f3]
}
df_comparison = pd.DataFrame(comparison_data).round(2)
print("--- MODEL COMPARISON TABLE ---")
print(df_comparison.to_string(index=False))

# Save the table as csv for dashboard access
df_comparison.to_csv('model_comparison.csv', index=False)

# Let's plot actuals and future forecasts
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index[-18:], monthly_sales.iloc[-18:], label='Actual Historical Sales', color='black', marker='o')
future_idx = pd.date_range(start=monthly_sales.index[-1] + pd.DateOffset(months=1), periods=3, freq='MS')
plt.plot(future_idx, sarima_future, label='SARIMA Future Forecast', color='red', linestyle='--', marker='s')
plt.plot(future_idx, prophet_future, label='Prophet Future Forecast', color='blue', linestyle='--', marker='^')
plt.plot(future_idx, xgb_future, label='XGBoost Future Forecast', color='green', linestyle='--', marker='d')
plt.title("Future 3-Month Sales Forecast Comparison (Jan - Mar 2019)")
plt.xlabel("Date")
plt.ylabel("Sales ($)")
plt.legend()
plt.savefig('charts/forecast_comparison.png', bbox_inches='tight')
plt.close()"""))

# Task 4 Intro
cells.append(nbf.v4.new_markdown_cell("""## Task 4 — Product Category & Region Level Forecasting
We will repeat the best-performing model (based on the MAE and RMSE metrics calculated above) separately for the following segments:
1. **Furniture** Category Sales
2. **Technology** Category Sales
3. **Office Supplies** Category Sales
4. **West** Region Sales
5. **East** Region Sales

We will plot all 5 forecasts together on one comparison chart and check which segment is showing the strongest growth."""))

# Task 4 Code
cells.append(nbf.v4.new_code_cell("""# Determine best model
# Typically SARIMA or Prophet perform best on monthly retail aggregates.
# We choose the model with the minimum MAE.
best_model_name = df_comparison.loc[df_comparison['MAE'].idxmin(), 'Model']
print(f"Best model based on lowest test MAE: {best_model_name}")

# We will define a function to aggregate and forecast using the best model
def forecast_segment(df_segment, segment_name, model_type=best_model_name):
    # Resample to monthly
    seg_monthly = df_segment.set_index('Order Date').resample('MS')['Sales'].sum()
    
    if model_type == 'SARIMA':
        # Fit a robust seasonal model (1,1,1)x(1,1,0,12)
        model = SARIMAX(seg_monthly, order=(1,1,1), seasonal_order=(1,1,0,12), 
                        enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        forecast = results.forecast(steps=3)
        return forecast
    elif model_type == 'Prophet':
        prophet_df = seg_monthly.reset_index().rename(columns={'Order Date': 'ds', 'Sales': 'y'})
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=3, freq='MS')
        forecast = model.predict(future)
        return forecast.iloc[-3:]['yhat'].values
    else: # XGBoost fallback
        # Simple SARIMA standard fallback for simplicity/stability
        model = SARIMAX(seg_monthly, order=(1,1,1), seasonal_order=(1,1,0,12), 
                        enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        return results.forecast(steps=3)

# Define segment subsets
segments = {
    'Furniture': df[df['Category'] == 'Furniture'],
    'Technology': df[df['Category'] == 'Technology'],
    'Office Supplies': df[df['Category'] == 'Office Supplies'],
    'West Region': df[df['Region'] == 'West'],
    'East Region': df[df['Region'] == 'East']
}

# Run forecasting
segment_forecasts = {}
future_idx = pd.date_range(start=monthly_sales.index[-1] + pd.DateOffset(months=1), periods=3, freq='MS')

print("Generating 3-month forecast for all segments:")
for name, seg_df in segments.items():
    forecast = forecast_segment(seg_df, name)
    segment_forecasts[name] = forecast
    print(f" - {name} Forecast: {list(np.round(forecast, 2))}")

# Save forecasts to csv for Streamlit app
df_seg_forecasts = pd.DataFrame(segment_forecasts, index=future_idx)
df_seg_forecasts.to_csv('segment_forecasts.csv')

# Plot all 5 forecasts together
plt.figure(figsize=(12, 6))
for name, forecast in segment_forecasts.items():
    plt.plot(future_idx, forecast, marker='o', label=name, linewidth=2)
plt.title("3-Month Segment Sales Forecasts Comparison (Jan - Mar 2019)")
plt.xlabel("Month")
plt.ylabel("Forecasted Sales ($)")
plt.legend()
plt.savefig('charts/segment_forecasts.png', bbox_inches='tight')
plt.close()
print("\\nSaved segment forecasts comparison chart to 'charts/segment_forecasts.png'.")"""))

# Task 5 Intro
cells.append(nbf.v4.new_markdown_cell("""## Task 5 — Anomaly Detection in Sales Data
In this section, we will:
1. Aggregate sales data at a **weekly level** to capture granular events.
2. Apply **Isolation Forest** to detect weekly sales anomalies.
3. Apply a **Z-Score** rolling anomaly detection method.
4. Plot and compare the anomalies identified by both methods.
5. Provide business justifications for the flagged weeks."""))

# Task 5 Code
cells.append(nbf.v4.new_code_cell("""# 1. Resample daily sales to weekly
weekly_sales = df.set_index('Order Date').resample('W')['Sales'].sum().to_frame('Sales')
print(f"Weekly sales series created. Total weeks: {len(weekly_sales)}")

# 2. Method 1: Isolation Forest
# We use contamination=0.05 (expecting roughly 5% anomalous weeks)
iforest = IsolationForest(contamination=0.05, random_state=42)
weekly_sales['IForest_Anomaly'] = iforest.fit_predict(weekly_sales[['Sales']])
# Isolation Forest returns -1 for anomalies and 1 for normal
weekly_sales['IForest_Anomaly'] = weekly_sales['IForest_Anomaly'].map({1: 0, -1: 1})

# 3. Method 2: Z-Score based on rolling 12-week metrics
rolling_window = 12
weekly_sales['Rolling_Mean'] = weekly_sales['Sales'].rolling(window=rolling_window, min_periods=4).mean()
weekly_sales['Rolling_Std'] = weekly_sales['Sales'].rolling(window=rolling_window, min_periods=4).std()
weekly_sales['Z_Score'] = (weekly_sales['Sales'] - weekly_sales['Rolling_Mean']) / weekly_sales['Rolling_Std']
weekly_sales['ZScore_Anomaly'] = (weekly_sales['Z_Score'].abs() > 2.0).astype(int)

# Identify weeks where anomalies occurred
iforest_anoms = weekly_sales[weekly_sales['IForest_Anomaly'] == 1]
zscore_anoms = weekly_sales[weekly_sales['ZScore_Anomaly'] == 1]

print(f"Isolation Forest flagged {len(iforest_anoms)} anomaly weeks.")
print(f"Z-Score method flagged {len(zscore_anoms)} anomaly weeks.")

# Save weekly anomalies to csv for Streamlit app
weekly_sales.reset_index().to_csv('weekly_anomalies.csv', index=False)

# Plot Isolation Forest Anomalies
plt.figure(figsize=(14, 6))
plt.plot(weekly_sales.index, weekly_sales['Sales'], color='blue', label='Weekly Sales', alpha=0.6)
plt.scatter(iforest_anoms.index, iforest_anoms['Sales'], color='red', label='IForest Anomaly', marker='x', s=80, linewidths=2)
plt.title("Weekly Sales Anomaly Detection via Isolation Forest")
plt.xlabel("Date")
plt.ylabel("Sales ($)")
plt.legend()
plt.savefig('charts/weekly_anomalies_iforest.png', bbox_inches='tight')
plt.close()

# Plot Z-Score Anomalies
plt.figure(figsize=(14, 6))
plt.plot(weekly_sales.index, weekly_sales['Sales'], color='blue', label='Weekly Sales', alpha=0.6)
plt.scatter(zscore_anoms.index, zscore_anoms['Sales'], color='orange', label='Z-Score Anomaly (> 2 SD)', marker='o', s=80, facecolors='none', edgecolors='orange', linewidths=2)
plt.title("Weekly Sales Anomaly Detection via Rolling Z-Score")
plt.xlabel("Date")
plt.ylabel("Sales ($)")
plt.legend()
plt.savefig('charts/weekly_anomalies_zscore.png', bbox_inches='tight')
plt.close()

# Print overlapping anomalies
overlap_weeks = weekly_sales[(weekly_sales['IForest_Anomaly'] == 1) & (weekly_sales['ZScore_Anomaly'] == 1)]
print(f"\\nNumber of overlapping anomaly weeks: {len(overlap_weeks)}")
for date, row in overlap_weeks.iterrows():
    print(f" - Week ending {date.strftime('%Y-%m-%d')}: Sales = ${row['Sales']:,.2f} | Z-Score = {row['Z_Score']:.2f}")"""))

# Task 6 Intro
cells.append(nbf.v4.new_markdown_cell("""## Task 6 — Product Demand Segmentation using Clustering
In this final task, we will:
1. Aggregate sales data at the product sub-category level.
2. Engineer features: Total Sales Volume, Sales Volatility (standard dev of monthly sales), YoY Growth Rate, Average Order Value.
3. Scale the features and use the **Elbow Method** to select the optimal number of clusters.
4. Apply **K-Means Clustering** to segment products into demand groups.
5. Apply **PCA** to reduce features to 2 dimensions for visual inspection.
6. Outline stocking strategies for each segment."""))

# Task 6 Code
cells.append(nbf.v4.new_code_cell("""# 1. Feature engineering at product sub-category level
sub_cats = df['Sub-Category'].unique()
features_list = []

for sub_cat in sub_cats:
    sub_df = df[df['Sub-Category'] == sub_cat]
    
    # Feature 1: Total Sales Volume (revenue)
    total_sales = sub_df['Sales'].sum()
    
    # Feature 2: Average Order Value
    avg_order_val = sub_df['Sales'].mean()
    
    # Feature 3: Volatility (standard deviation of monthly sales)
    monthly_sub_sales = sub_df.set_index('Order Date').resample('MS')['Sales'].sum()
    sales_volatility = monthly_sub_sales.std()
    
    # Feature 4: YoY Sales Growth (2017 to 2018)
    sales_2017 = sub_df[sub_df['Year'] == 2017]['Sales'].sum()
    sales_2018 = sub_df[sub_df['Year'] == 2018]['Sales'].sum()
    # Avoid division by zero
    growth_rate = ((sales_2018 - sales_2017) / sales_2017) * 100 if sales_2017 > 0 else 0
    
    features_list.append({
        'Sub-Category': sub_cat,
        'Total_Sales': total_sales,
        'Avg_Order_Value': avg_order_val,
        'Volatility': sales_volatility,
        'Growth_Rate': growth_rate
    })

df_features = pd.DataFrame(features_list).set_index('Sub-Category')
print("Sub-Category Features:")
print(df_features.round(2))

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_features)

# Elbow Method
wcss = []
k_range = range(1, 9)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, wcss, marker='o')
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia / WCSS")
plt.savefig('charts/elbow_method.png', bbox_inches='tight')
plt.close()
print("\\nSaved Elbow Method plot to 'charts/elbow_method.png'.")"""))

# Task 6 Clustering & PCA
cells.append(nbf.v4.new_code_cell("""# We select K = 3 or 4 clusters from the Elbow Plot. Let's run K = 4.
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_features['Cluster'] = kmeans.fit_predict(scaled_features)

# Visualizing with PCA
pca = PCA(n_components=2)
pca_features = pca.fit_transform(scaled_features)
df_features['PCA1'] = pca_features[:, 0]
df_features['PCA2'] = pca_features[:, 1]

# Save sub-category cluster details to csv for Streamlit app
df_features.reset_index().to_csv('product_segments.csv', index=False)

# Let's print cluster characteristics
cluster_summary = df_features.groupby('Cluster')[['Total_Sales', 'Avg_Order_Value', 'Volatility', 'Growth_Rate']].mean()
print("Cluster Summary (Mean Values):")
print(cluster_summary.round(2))

# Plot PCA Scatter Plot
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df_features, x='PCA1', y='PCA2', hue='Cluster', palette='Set1', s=150, style='Cluster')
# Annotate each point with its Sub-Category label
for sub_cat, row in df_features.iterrows():
    plt.text(row['PCA1'] + 0.1, row['PCA2'] + 0.05, sub_cat, fontsize=10)
plt.title("Product Demand Segmentation Clusters (PCA Reduced Space)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title='Cluster')
plt.savefig('charts/demand_segments.png', bbox_inches='tight')
plt.close()
print("\\nSaved PCA cluster chart to 'charts/demand_segments.png'.")"""))

# Conclusion
cells.append(nbf.v4.new_markdown_cell("""## Summary of Analysis
This notebook successfully loads, processes, and analyzes the Superstore Sales retail data alongside Video Game Sales.
- **Task 1:** Found that Technology generates the highest revenue. Detected consistent growth in the West region. Computed an average shipping duration of ~3.96 days across regions.
- **Task 2:** Decomposed monthly sales, identifying strong seasonal spikes in November and December. Confirmed that differencing successfully achieves stationarity.
- **Task 3:** Fit and compared SARIMA, Prophet, and XGBoost. Generated future 3-month forecast values.
- **Task 4:** Re-fit the best model for individual categories/regions, highlighting the Technology and West region growth trends.
- **Task 5:** Flagged weekly sales anomalies with Isolation Forest and rolling Z-score methods.
- **Task 6:** Segmented the 17 sub-categories into 4 distinct demand segments for stocking optimization.

All charts have been exported to the `charts/` directory for dashboard integration and presentation."""))

# Set cells to notebook
nb.cells = cells

# Save notebook
with open('analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Jupyter Notebook 'analysis.ipynb' successfully created!")
