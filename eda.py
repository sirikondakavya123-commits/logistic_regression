# IMPORT LIBRARIES

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

# LOAD DATASET

df = pd.read_csv("online_shoppers_intention.csv")

# REMOVE EXTRA SPACES

df.columns = df.columns.str.strip()

print("Dataset Loaded Successfully")

print(df.head())

# BASIC INFO

print("\nShape of Dataset:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nInfo:")
df.info()

print("\nDescription:")
print(df.describe())

# CHECK MISSING VALUES

missing = df.isnull().sum()

missing = missing[missing > 0]

print("\nMissing Values:")

print(missing.sort_values(ascending=False))

# HANDLE MISSING VALUES

# NUMERICAL COLUMNS

num_cols = df.select_dtypes(include=np.number).columns

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# CATEGORICAL COLUMNS

cat_cols = df.select_dtypes(include='object').columns

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After Handling:")

print(df.isnull().sum().sum())

# CHECK DUPLICATES

print("\nDuplicate Rows:")

print(df.duplicated().sum())

# REMOVE DUPLICATES

df = df.drop_duplicates()

print("\nShape After Removing Duplicates:")

print(df.shape)

# REMOVE OUTLIERS USING IQR
# USING ProductRelated_Duration COLUMN

Q1 = df["ProductRelated_Duration"].quantile(0.25)

Q3 = df["ProductRelated_Duration"].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR

upper = Q3 + 1.5 * IQR

df = df[
    (df["ProductRelated_Duration"] >= lower) &
    (df["ProductRelated_Duration"] <= upper)
]

print("\nShape After Removing Outliers:")

print(df.shape)

# LABEL ENCODING

encoder = LabelEncoder()

for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

print("\nCategorical Encoding Completed")

# TARGET COLUMN CONVERSION

df['Revenue'] = df['Revenue'].astype(int)

print("\nTarget Column Converted")

# CORRELATION

print("\nCorrelation Matrix:")

print(df.corr())

# SAVE CLEANED DATASET

df.to_csv('cleaned_online_shoppers.csv', index=False)

print("\nCleaned Dataset Saved Successfully")