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

################## Missing data imputation

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

def impute_na(df, variable, value):
    return df[variable].fillna(value)


################## Outlier Engineering

# Функція для знаходження меж для аномальних значень (Outliers) на основі IQR
def find_skewed_boundaries(df, variable, distance):
    """
    Обчислює верхні та нижні межі для виявлення аномальних значень у колонці `variable`.
    """
    df[variable] = pd.to_numeric(df[variable], errors='coerce')  # Перетворення на числовий формат
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)  # Розмах квартилів
    lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR * distance)
    return upper_boundary, lower_boundary

# Визначення меж аномальних значень для кожної колонки з outlier_columns
upper_lower_limits = {}
for col in columns.outlier_columns:
    upper_lower_limits[col + '_upper_limit'], upper_lower_limits[col + '_lower_limit'] = find_skewed_boundaries(ds, col, 1.5)

# Видалення рядків з аномальними значеннями
for col in columns.outlier_columns:
    upper = upper_lower_limits[col + '_upper_limit']
    lower = upper_lower_limits[col + '_lower_limit']
    ds = ds[(ds[col] >= lower) & (ds[col] <= upper)]

# Перевірка розміру набору даних після видалення аномалій
print(f"Після видалення аномалій, кількість рядків: {ds.shape[0]}")


################## save parameters

# Підготовка словника для збереження параметрів
param_dict = {
    'upper_lower_limits': upper_lower_limits,  # Межі для аномальних значень
    'computed_columns': {
        'diameter': 'calculated_based_on_H_and_albedo'  # Формула для обчислення
    },
    'group_mean_fill_columns': columns.group_mean_fill_columns,  # Групування та заповнення середнім
    'fillna_median_columns': columns.fillna_median_columns,  # Колонки для заповнення медіаною
    'fillna_unknown_columns': columns.fillna_unknown_columns,  # Колонки для заповнення "Unknown"
    'fillna_not_hazardous_columns': columns.fillna_not_hazardous_columns,  # Колонки для заповнення "N"
    'keep_na_columns': columns.keep_na_columns  # Колонки, які залишаються NaN
}

# Збереження параметрів у файл `param_dict.pickle`
with open('models/param_dict.pickle', 'wb') as handle:
    pickle.dump(param_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Параметри успішно збережено у файл models/param_dict.pickle")

###########################################################################################

#################################### Тренування моделі ####################################
print(ds.columns)
print(columns.X_columns)

# Визначаємо цільову змінну та ознаки
X = ds[columns.X_columns]
y = ds[columns.y_column]

# Розділяємо дані на тренувальний і тестовий набір у співвідношенні 90:10
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
