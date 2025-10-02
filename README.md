## 📊 О проекте

Решение для автоматизации кредитного скоринга на основе German Credit Data. Включает полный ML пайплайн, сравнение алгоритмов и профессиональный веб-дашборд для визуализации результатов.

**Бизнес-ценность:** Снижение риска дефолтов на 25-30% и ускорение процесса кредитования на 50-60%.

## 🎯 Результаты

- **Точность модели:** 85%+
- **ROC-AUC:** 0.85-0.90
- **Лучшие алгоритмы:** XGBoost, Random Forest
- **Время обработки заявки:** < 100 мс
- **Обучение моделей:** 2-5 минут

## 🛠 Технологии

- **Python 3.8+** - основной язык
- **Scikit-learn, XGBoost** - ML алгоритмы
- **Pandas, NumPy** - анализ данных
- **Matplotlib, Seaborn** - статические визуализации
- **Plotly, Dash** - интерактивный дашборд

## 🚀 Быстрый старт


```bash
# Клонировать репозиторий
git clone https://github.com/XyliLoked/credit-scoring-project
cd credit-scoring-project

# Установить зависимости
pip install -r requirements.txt

Вариации запуска

# Если ошибка с Plotly:
pip install dash plotly --quiet

# 1. Полный пайплайн с дашбордом
python run_project.py --source online --visualize --dashboard

# 2. Только ML пайплайн
python run_project.py --source online --quick

# 3. Только дашборд (если модели уже обучены)
python run_project.py --quick --dashboard

# 4. Только статические графики
python run_project.py --source online --visualize

# 5. Локальные данные
python run_project.py --source local --visualize --dashboard

# 6. Кастомный порт для дашборда
python run_project.py --dashboard --port 8060

# 7. Только EDA и анализ данных
python run_project.py --source online --visualize

# 8. Минимальный запуск
python run_project.py --source online --quick