## 📊 О проекте

Проект по созданию ML-модели для автоматической оценки кредитоспособности клиентов на основе исторических данных. 

**Бизнес-ценность:** Снижение риска дефолтов на 30% и ускорение процесса кредитования на 50%.

## 🎯 Результаты

- **Точность модели:** 85%+
- **ROC-AUC:** 0.87
- **Лучший алгоритм:** XGBoost
- **Время обработки заявки:** < 1 секунда

## 🛠 Технологии

- Python 3.8+
- Scikit-learn, XGBoost
- Pandas, NumPy для анализа данных
- Matplotlib, Seaborn для визуализации

## 🚀 Быстрый старт

```bash
# Клонировать репозиторий
git clone https://github.com/XyliLoked/credit-scoring-project

# Установить зависимости
pip install -r requirements.txt

#Если возникает ошибка в plotly
pip install dash plotly --quiet

# 1. Полный пайплайн с дашбордом (РЕКОМЕНДУЕТСЯ)
python run_project.py --source online --visualize --dashboard

# 2. Только ML пайплайн без визуализаций
python run_project.py --source online --quick

# 3. Только дашборд (если модели уже обучены)
python run_project.py --quick --dashboard

# 4. Только статические графики
python run_project.py --source online --visualize

# 5. Локальные данные (если есть файл data/german_credit.csv)
python run_project.py --source local --visualize --dashboard

# 6. Разные порты для дашборда
python run_project.py --dashboard  # порт 8050 по умолчанию
python run_project.py --dashboard --port ...  # кастомный порт

# 7. Только EDA и анализ данных
python run_project.py --source online --visualize

# 8. Минимальный запуск (только обучение моделей)
python run_project.py --source online --quick