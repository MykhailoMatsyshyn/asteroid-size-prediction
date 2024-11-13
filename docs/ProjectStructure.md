# **Прогнозування розміру астероїдів**

## **Структура проекту**

Проект для прогнозування розміру астероїдів з метою оцінки потенційної небезпеки для Землі.

## **Дерево каталогу**

```
RGR
├─ .git
├─ .gitignore
│
├─ configs
│  ├─ columns.py
│  ├─ model_best_hyperparameters.py
│  └─ __pycache__
│     ├─ columns.cpython-311.pyc
│     └─ model_best_hyperparameters.cpython-311.pyc
│
├─ data
│  ├─ comparison_table.csv
│  ├─ new_data.csv
│  ├─ new_data_with_actual.csv
│  ├─ prediction_results.csv
│  ├─ prime_dataset.csv
│  └─ train_data.csv
│
├─ docs
│  └─ ProjectStructure.md
│
├─ models
│  ├─ finalized_model.sav
│  ├─ logs
│  │  ├─ prediction_result
│  │  │  ├─ deviation_scatter_plot.png
│  │  │  └─ prediction_result.png
│  │  └─ train_model
│  │     ├─ cooks_distance.png
│  │     ├─ error_distribution.png
│  │     ├─ evaluation_metrics.txt
│  │     ├─ feature_importances.csv
│  │     ├─ feature_importances.png
│  │     ├─ learning_curve.png
│  │     ├─ residual_plot.png
│  │     └─ y_test_vs_y_pred.png
│  └─ param_dict.pickle
│
├─ pipeline
│  ├─ predict.py
│  └─ train.py
│
├─ README.md
├─ requirements.txt
│
├─ src
│  ├─ 1_DataAnalysis.ipynb
│  ├─ 2_DataProcessing.ipynb
│  ├─ 3_ModelTraining.ipynb
│  ├─ filtered_dataset.csv
│  └─ processed_dataset.csv
│
└─ utils
   ├─ analyze_results.py.ipynb
   ├─ plotting_utils.py
   ├─ utils for dataset split and save.ipynb
   └─ __pycache__
      └─ plotting_utils.cpython-311.pyc
```

## **Опис каталогів і файлів**

- **configs/**: Файли конфігурацій, які містять параметри для інженерії ознак і налаштування моделі.

  - `columns.py`: Містить інформацію про використані колонки у наборі даних.
  - `model_best_hyperparameters.py`: Гіперпараметри, використані для тренування моделі.
  - `__pycache/`: Каталог з компільованими Python файлами.

- **data/**: Містить навчальні, тестові дані, а також файли з результатами.

  - `comparison_table.csv`: Таблиця для порівняння фактичних і передбачених значень.
  - `new_data.csv`: Нові дані для передбачення.
  - `new_data_with_actual.csv`: Нові дані з доданими фактичними значеннями для порівняння.
  - `prediction_results.csv`: Файл з результатами передбачення.
  - `prime_dataset.csv`: Початковий набір даних з інформацією про астероїди.
  - `train_data.csv`: Дані для тренування.

- **docs/**: Документація проекту.

  - `ProjectStructure.md`: Документація щодо структури проекту.

- **models/**: Містить треновану модель та результати.

  - `logs/`: Логи тренування моделі та візуалізації.
    - `prediction_result/`: Логи для результатів передбачення.
      - `deviation_scatter_plot.png`: Графік відхилень між передбаченим і фактичним діаметром.
      - `prediction_result.png`: Графік результатів передбачення.
    - `train_model/`: Логи для тренування моделі.
      - `evaluation_metrics.txt`: Метрики оцінки моделі.
      - `feature_importances.csv`: CSV файл з важливістю ознак.
      - `learning_curve.png`: Крива навчання.
      - `residual_plot.png`: Графік залишків.
  - `finalized_model.sav`: Файл з натренованою моделлю.
  - `param_dict.pickle`: Параметри для обробки даних.

- **pipeline/**: Скрипти для тренування і передбачення моделі.

  - `train.py`: Скрипт для тренування моделі.
  - `predict.py`: Скрипт для передбачення на нових даних.

- **README.md**: Основна документація проекту з інформацією про використання, вимоги, і установку.

- **requirements.txt**: Файл із залежностями проекту для швидкої установки необхідних бібліотек.

- **src/**: Скрипти та файли для аналізу даних і тренування моделі.

  - `1_DataAnalysis.ipynb`: Jupyter Notebook для попереднього аналізу даних.
  - `2_DataProcessing.ipynb`: Jupyter Notebook для попередньої обробки даних.
  - `3_ModelTraining.ipynb`: Jupyter Notebook для тренування моделі.
  - `filtered_dataset.csv`: Очищений набір даних.
  - `processed_dataset.csv`: Оброблений набір даних.

- **utils/**: Утилітні файли та скрипти.
  - `analyze_results.py.ipynb`: Jupyter Notebook для аналізу результатів передбачень.
  - `plotting_utils.py`: Скрипт з функціями для візуалізації даних.
  - `utils for dataset split and save.ipynb`: Jupyter Notebook з функціями для поділу і збереження даних.
  - `__pycache/`: Каталог з компільованими Python файлами.
