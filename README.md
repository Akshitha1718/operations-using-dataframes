# operations-using-dataframes
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.DataFrame({
    'Date': ['2023-01-01', '02-01-2023', 'Invalid'],
    'Category': ['Apple', 'apple', 'Banana'],
    'Value': [10, 1500, 20],
    'Missing': [1, np.nan, 3]
})

print("Original Data:\n", df)

df.drop_duplicates(inplace=True)
df['Missing'].fillna(df['Missing'].mean(), inplace=True)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce').fillna(method='ffill')
df['Category'] = df['Category'].str.lower()
df['Value'] = np.where(df['Value'] > 100, 100, df['Value'])
scaler = MinMaxScaler()
df['Value_norm'] = scaler.fit_transform(df[['Value']])

print("\nCleaned Data:\n", df)


