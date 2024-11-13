import sys
import os
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs import columns

# Завантаження даних для передбачення
ds = pd.read_csv("data/new_data.csv")

# Завантаження даних для передбачення
ds = pd.read_csv("data/new_data.csv")

# Перевірка на порожній набір даних
if ds.empty:
    print("Помилка: Набір вхідних даних порожній. Будь ласка, надайте коректні дані для передбачення.")
    sys.exit(1)

print(f"Розмір нових даних: {ds.shape}")

# Завантаження параметрів
param_dict = pickle.load(open('models/param_dict.pickle', 'rb'))

#################################### Feature Engineering ####################################

# Імпутація відсутніх даних
def impute_na(df, variable, value):
    return df[variable].fillna(value)

# 1. Видалення непотрібних колонок
ds = ds.drop(columns=columns.columns_to_drop, errors='ignore')

# 2. Імпутація пропущених значень
for col in columns.fillna_median_columns:
    if col in ds.columns:
        ds[col] = ds[col].fillna(param_dict['fillna_median_values'][col])
    else:
        print(f"Попередження: Колонка '{col}' відсутня у вхідних даних.")

for target_col, group_cols in param_dict['group_mean_fill_columns'].items():
    ds[target_col] = ds.groupby(group_cols)[target_col].transform(lambda x: x.fillna(x.mean()))

print(ds)

# 3. Обробка викидів
for column in columns.outlier_columns:
    ds[column] = ds[column].astype(float)
    ds = ds[~ np.where(ds[column] > param_dict['upper_lower_limits'][column+'_upper_limit'], True,
                       np.where(ds[column] < param_dict['upper_lower_limits'][column+'_lower_limit'], True, False))]

#################################### Передбачення ####################################

# Визначення ознак
X = ds[columns.X_columns]

# Завантаження моделі
model = pickle.load(open('models/finalized_model.sav', 'rb'))

# Передбачення
ds['diameter_pred'] = model.predict(X)

# Збереження результатів
output_file = "data/prediction_results.csv"
ds.to_csv(output_file, index=False)

print(f"Передбачення завершено. Результати збережено в {output_file}")