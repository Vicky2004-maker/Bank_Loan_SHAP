import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.compose import ColumnTransformer

# %%

prev_data = pd.read_csv(r"S:\Dataset\Bank Loan\previous_application.csv")
data = pd.read_csv(r"S:\Dataset\Bank Loan\application_data.csv")

# %% Dropping the columns with more than 48% of null values

na_sum = data.isna().sum()
na_sum = na_sum[na_sum != 0]
na_sum = (na_sum / len(data)) * 100
na_sum = na_sum[na_sum > 48.0]

data.drop(na_sum.index, axis=1, inplace=True)
print(
    f'Dropped {len(na_sum.index)} columns from data\nThese {len(na_sum.index)} columns had more than 48% of null values')

# %% Retrieving the Categorical Columns (object dtype)

categorical_columns = data.columns[data.dtypes == object]

print(data[categorical_columns])

# %% Visualizing the number of unique values in each categorical columns

unique_cat_values = []
unique_cat_length = []

for i in categorical_columns:
    __uniq = list(data[i].unique())
    unique_cat_values.append(__uniq)
    unique_cat_length.append(len(__uniq))

plt.figure(figsize=(12, 10))
plt.title('Number of unique values in each categorical columns')
plt.bar(categorical_columns, unique_cat_length, width=0.5, color='red')
plt.yticks(np.arange(0, 61, 2))
plt.grid(axis='y')
plt.xticks(rotation=20, rotation_mode='anchor', ha='right')
plt.show()

# %%

filter_2 = data[categorical_columns].isna().sum()

print((filter_2 != 0))
