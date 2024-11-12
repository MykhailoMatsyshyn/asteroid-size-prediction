import sys
import os

import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs import columns
from configs import model_best_hyperparameters

# Завантаження даних для тренування
ds = pd.read_csv("data/train_data.csv")

#################################### feature engineering ####################################

# 1. Видалення колонок без корисної інформації
ds = ds.drop(columns=columns.columns_to_drop, errors='ignore')

# 2. Заповнення значень 'Unknown' у вказаних колонках
def impute_unknown(df, columns):
    for col in columns:
        df[col] = df[col].fillna('Unknown')
    return df

ds = impute_unknown(ds, columns.fillna_unknown_columns)

# 3. Заповнення значенням 'N' для не небезпечних астероїдів
def impute_not_hazardous(df, columns):
    for col in columns:
        df[col] = df[col].fillna('N')
    return df

ds = impute_not_hazardous(ds, columns.fillna_not_hazardous_columns)

# 4. Заповнення медіанним значенням
def impute_median(df, columns):
    for col in columns:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)
    return df

ds = impute_median(ds, columns.fillna_median_columns)

# 5. Заповнення середнім значенням на основі групування
def group_mean_fill(df, group_columns_dict):
    for target_col, group_cols in group_columns_dict.items():
        df[target_col] = df.groupby(group_cols)[target_col].transform(lambda x: x.fillna(x.mean()))
    return df

ds = group_mean_fill(ds, columns.group_mean_fill_columns)

# 6. Обчислення значень на основі формул
def compute_values(df, computed_columns):
    if 'diameter' in computed_columns:
        # Обчислення `diameter` на основі формули, якщо доступні `H` та `albedo`
        df['diameter'] = df.apply(
            lambda row: 1329 / (row['albedo']**0.5) * (10**(-0.2 * row['H'])) 
            if pd.notna(row['H']) and pd.notna(row['albedo']) else row['diameter'],
            axis=1
        )
    return df

ds = compute_values(ds, columns.computed_columns)

# 7. Залишаємо значення NaN для заданих колонок
# (нічого не змінюємо для `keep_na_columns` — залишаємо пропущені значення)

#############################################################################################

# Перевіряємо результати попередньої обробки
print(ds.info())