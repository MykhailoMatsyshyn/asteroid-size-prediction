import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import learning_curve

def plot_actual_vs_predicted(y_test, y_pred, save_path):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
    plt.title("Прогнозовані значення vs Фактичні")
    plt.xlabel("Фактичні значення")
    plt.ylabel("Прогнозовані значення")
    plt.savefig(save_path)
    plt.close()

def plot_residuals(y_test, y_pred, save_path):
    # Перетворення у масиви
    y_test = y_test.values.ravel() if hasattr(y_test, 'values') else y_test
    y_pred = np.array(y_pred)

    # Перевірка форм
    print(f"Shape of y_test: {y_test.shape}")
    print(f"Shape of y_pred: {y_pred.shape}")

    # Обчислення залишків
    residuals = y_test - y_pred
    print("Residuals calculated successfully.")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.title("Графік залишків (Residual Plot)")
    plt.xlabel("Прогнозовані значення")
    plt.ylabel("Залишки")
    plt.savefig(save_path)
    plt.close()

def plot_feature_importances(feature_importances, feature_names, save_path):
    feature_importances_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    feature_importances_df.to_csv(save_path.replace(".png", ".csv"), index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances_df["Feature"], feature_importances_df["Importance"], color='skyblue')
    plt.xlabel("Важливість")
    plt.title("Важливість ознак (XGBoost)")
    plt.gca().invert_yaxis()
    plt.savefig(save_path)
    plt.close()

def plot_cooks_distance(y_test, y_pred, save_path):
    model = sm.OLS(y_test, sm.add_constant(y_pred)).fit()
    influence = model.get_influence()
    cooks = influence.cooks_distance[0]

    plt.figure(figsize=(8, 6))
    plt.stem(np.arange(len(cooks)), cooks, markerfmt=",", basefmt=" ")
    plt.title("Cooks Distance")
    plt.xlabel("Індекс")
    plt.ylabel("Вплив")
    plt.savefig(save_path)
    plt.close()

def plot_learning_curve(estimator, X, y, save_path):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring='neg_mean_squared_error'
    )
    train_scores_mean = -train_scores.mean(axis=1)
    test_scores_mean = -test_scores.mean(axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_scores_mean, label='Тренувальний набір', color='blue')
    plt.plot(train_sizes, test_scores_mean, label='Тестовий набір', color='red')
    plt.xlabel('Кількість навчальних даних')
    plt.ylabel('MSE')
    plt.title('Крива навчання')
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()

def plot_error_distribution(errors, save_path):
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, kde=True, bins=30, color='orange')
    plt.xlabel('Абсолютна помилка')
    plt.ylabel('Частота')
    plt.title('Розподіл помилок')
    plt.savefig(save_path)
    plt.close()
