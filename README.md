# Ex.No: 6               HOLT WINTERS METHOD
### Date: 



### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```
# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. Load the existing dataset from the correct local path
target_path = '/content/superstore_final_dataset.csv'

if os.path.exists(target_path):
    print(f'Loading existing dataset from: {target_path}')
    data = pd.read_csv(target_path, encoding='cp1252')
else:
    print('Dataset file not found at expected path. Generating synthetic Superstore dataset instead.')
    np.random.seed(42)
    date_range = pd.date_range(start='2018-01-01', end='2021-12-31', freq='D')
    base_sales = np.random.normal(loc=200, scale=50, size=len(date_range))
    seasonal_effect = 100 * np.sin(2 * np.pi * date_range.dayofyear / 365.25)
    trend_effect = np.linspace(0, 150, len(date_range))
    sales = np.clip(base_sales + seasonal_effect + trend_effect, 10, None)
    data = pd.DataFrame({
        'Order_Date': date_range,
        'Sales': sales
    })
    data.to_csv(target_path, index=False)

# Map columns to match standard expected structure
if 'Order Date' in data.columns:
    data.rename(columns={'Order Date': 'Order_Date'}, inplace=True)
if 'Sales' not in data.columns and 'sales' in data.columns:
    data.rename(columns={'sales': 'Sales'}, inplace=True)

data['Order_Date'] = pd.to_datetime(
    data['Order_Date'], format='mixed', dayfirst=False
)

data = data.set_index('Order_Date')

print(data.head())

# 2. Resample and plot data
data_monthly = data['Sales'].resample('MS').sum()

print(data_monthly.head())

data_monthly.plot(figsize=(10,5))
plt.title('Monthly Superstore Sales')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.show()

# 3. Scale the data and check for seasonality (Using a strictly positive range [1, 2] to prevent multiplicative HW errors)
scaler = MinMaxScaler(feature_range=(1, 2))

scaled_data = pd.Series(
    scaler.fit_transform(
        data_monthly.values.reshape(-1, 1)
    ).flatten(),
    index=data_monthly.index
)

scaled_data.plot(figsize=(10,5))
plt.title('Scaled Superstore Sales')
plt.xlabel('Month')
plt.ylabel('Scaled Sales')
plt.show()

# 4. Decomposition plot
decomposition = seasonal_decompose(
    data_monthly,
    model='additive',
    period=12
)
decomposition.plot()
plt.show()

# 5. Split train and test data
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
ttest_data = scaled_data[int(len(scaled_data) * 0.8):]

# 6. Create Holt-Winters model
model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

# 7. Test prediction
test_predictions_add = model_add.forecast(
    steps=len(ttest_data)
)

# 8. Plot test prediction
ax = train_data.plot(figsize=(10,5))
test_predictions_add.plot(ax=ax)
ttest_data.plot(ax=ax)
ax.legend([
    'train_data',
    'test_predictions_add',
    'test_data'
])
ax.set_title('Visual Evaluation')
plt.show()

# 9. Calculate RMSE
rmse = np.sqrt(
    mean_squared_error(ttest_data, test_predictions_add)
)
print("RMSE:", rmse)
print("Standard Deviation and Mean:")
print(
    np.sqrt(scaled_data.var()),
    scaled_data.mean()
)

# 10. Create final model and predict future sales
final_model = ExponentialSmoothing(
    data_monthly,
    trend='add',
    seasonal='mul',
    seasonal_periods=12
).fit()

# Predict next 12 months
final_predictions = final_model.forecast(steps=12)

# 11. Plot final prediction
ax = data_monthly.plot(figsize=(10,5))
final_predictions.plot(ax=ax)
ax.legend([
    'Monthly Sales',
    'Final Predictions'
])
ax.set_xlabel('Date')
ax.set_ylabel('Sales')
ax.set_title('Superstore Sales Prediction')
plt.show()

# 12. Display predicted sales
print("Future 12 Months Sales Prediction:")
print(final_predictions)
```
## output

<img width="770" height="778" alt="Screenshot 2026-11-21 090826" src="https://github.com/user-attachments/assets/b6dddac9-4904-4bcc-8714-dfa9fe50acfe" />

<img width="846" height="486" alt="image" src="https://github.com/user-attachments/assets/a7916dcb-59fe-433b-9a1d-36849f8efb01" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
