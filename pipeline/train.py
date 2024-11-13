import sys
import os

import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
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


################## Save Parameters

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
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)

# Ініціалізація моделі XGBRegressor із найкращими гіперпараметрами
model = XGBRegressor(**model_best_hyperparameters.params, random_state=42)
model.fit(X_train, y_train)

# Передбачення на тестових даних
y_pred = model.predict(X_test)

# Обчислення метрик
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Збереження метрик у файл
os.makedirs("models/logs", exist_ok=True)
with open("models/logs/evaluation_metrics.txt", "w") as f:
    f.write(f"Mean Squared Error (MSE): {mse}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
    f.write(f"Mean Absolute Error (MAE): {mae}\n")
    f.write(f"R-squared (R^2): {r2}\n")


#################################### Графіки ####################################

from utils.plotting_utils import (
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importances,
    plot_cooks_distance,
    plot_learning_curve,
    plot_error_distribution
)


# 1. Графік прогнозованих значень проти фактичних
plot_actual_vs_predicted(y_test, y_pred, "models/logs/y_test_vs_y_pred.png")

# 2. Графік залишків
plot_residuals(y_test, y_pred, "models/logs/residual_plot.png")

# 3. Важливість ознак
plot_feature_importances(model.feature_importances_, columns.X_columns, "models/logs/feature_importances.png")

# 4. Cook's Distance
plot_cooks_distance(y_test, y_pred, "models/logs/cooks_distance.png")

# 5. Крива навчання
plot_learning_curve(XGBRegressor(**model_best_hyperparameters.params), X, y, "models/logs/learning_curve.png")

# 6. Розподіл помилок
# errors = abs(y_test - y_pred)
# plot_error_distribution(errors, "models/logs/error_distribution.png")

# # 6. Розподіл помилок
# errors = abs(y_test - y_pred)

# plt.figure(figsize=(8, 6))
# sns.histplot(errors, kde=True, bins=30, color='orange')
# plt.xlabel('Абсолютна помилка')
# plt.ylabel('Частота')
# plt.title('Розподіл помилок')
# plt.savefig('models/logs/error_distribution.png')  # Збереження графіка
# plt.close()

# Збереження моделі
filename = 'models/finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

###########################################################################################
