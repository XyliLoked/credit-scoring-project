# credit_scoring.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import warnings
import os
import sys
from datetime import datetime

warnings.filterwarnings('ignore')

# 🔧 ПРОВЕРКА PLOTLY
try:
    import plotly
    PLOTLY_AVAILABLE = True
    print("✅ Plotly доступен")
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly не установлен")
    
class CreditScoringModel:
    """
    Класс для построения модели кредитного скоринга
    """
    
    def __init__(self):
        self.df = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, source='online'):
        """
        Загрузка данных из различных источников
        """
        print("📊 ЗАГРУЗКА ДАННЫХ...")
        print("-" * 50)
        
        if source == 'online':
            # Загрузка с UCI
            try:
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
                column_names = [
                    'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
                    'savings_account', 'employment', 'installment_rate', 'personal_status_sex',
                    'other_debtors', 'present_residence', 'property', 'age', 'other_installment_plans',
                    'housing', 'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker', 'credit_risk'
                ]
                self.df = pd.read_csv(url, delim_whitespace=True, names=column_names, header=None)
                print("✅ German Credit Dataset успешно загружен с UCI!")
                
            except Exception as e:
                print(f"❌ Ошибка загрузки с UCI: {e}")
                self._create_sample_data()
                
        elif source == 'local':
            # Загрузка локального файла
            try:
                self.df = pd.read_csv('data/german_credit.csv')
                print("✅ Данные загружены из локального файла!")
            except:
                print("❌ Локальный файл не найден. Создаем демо-данные...")
                self._create_sample_data()
        
        # Преобразование целевой переменной
        if 'credit_risk' in self.df.columns:
            self.df['target'] = self.df['credit_risk'].map({1: 0, 2: 1})
            print("✅ Целевая переменная преобразована")
        
        print(f"📊 Размер датасета: {self.df.shape}")
        print(f"🎯 Распределение классов:")
        print(self.df['target'].value_counts())
        
        return self.df
    
    def _create_sample_data(self):
        """Создание демо-данных если загрузка не удалась"""
        print("🔄 Создание демо-датасета...")
        np.random.seed(42)
        n_samples = 1000
        
        self.df = pd.DataFrame({
            'checking_account': np.random.choice(['A11', 'A12', 'A13', 'A14'], n_samples),
            'duration': np.random.randint(6, 72, n_samples),
            'credit_history': np.random.choice(['A30', 'A31', 'A32', 'A33', 'A34'], n_samples),
            'credit_amount': np.random.randint(250, 18424, n_samples),
            'savings_account': np.random.choice(['A61', 'A62', 'A63', 'A64', 'A65'], n_samples),
            'employment': np.random.choice(['A71', 'A72', 'A73', 'A74', 'A75'], n_samples),
            'age': np.random.randint(19, 75, n_samples),
            'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        })
        print("✅ Демо-датасет создан")
    
    def explore_data(self):
        """
        Анализ и визуализация данных
        """
        print("\n📊 АНАЛИЗ ДАННЫХ...")
        print("-" * 50)
        
        # Базовая информация
        print("📋 Информация о данных:")
        print(self.df.info())
        
        print("\n📈 Статистика числовых признаков:")
        print(self.df.describe())
        
        # Создание визуализаций
        self._create_visualizations()
        
    def _create_visualizations(self):
        """Создание графиков для EDA"""
        print("📊 Создание визуализаций...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('АНАЛИЗ КРЕДИТНЫХ ДАННЫХ', fontsize=16, fontweight='bold')
        
        # 1. Распределение целевой переменной
        target_counts = self.df['target'].value_counts()
        axes[0,0].pie(target_counts.values,
                    labels=[f'Хорошие ({target_counts[0]})', f'Плохие ({target_counts[1]})'],
                    autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'], startangle=90)
        axes[0,0].set_title('Распределение классов клиентов', fontweight='bold')
        
        # 2. Распределение суммы кредита
        if 'credit_amount' in self.df.columns:
            axes[0,1].hist(self.df['credit_amount'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0,1].set_xlabel('Сумма кредита (DM)')
            axes[0,1].set_ylabel('Частота')
            axes[0,1].set_title('Распределение суммы кредита', fontweight='bold')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Распределение возраста
        if 'age' in self.df.columns:
            axes[1,0].hist(self.df['age'], bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[1,0].set_xlabel('Возраст')
            axes[1,0].set_ylabel('Частота')
            axes[1,0].set_title('Распределение возраста', fontweight='bold')
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Корреляционная матрица
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 1:
            correlation_matrix = self.df[numeric_columns].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,1],
                    fmt='.2f', annot_kws={'size': 8})
            axes[1,1].set_title('Корреляционная матрица', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('eda_visualization.png', dpi=300, bbox_inches='tight')
        print("✅ Визуализации сохранены в 'eda_visualization.png'")
        plt.show()
    
    def preprocess_data(self):
        """
        Предобработка и feature engineering
        """
        print("\n🔧 ПРЕДОБРАБОТКА ДАННЫХ...")
        print("-" * 50)
        
        df_processed = self.df.copy()
        
        # Удаляем целевую переменную
        y = df_processed['target']
        df_processed = df_processed.drop('target', axis=1)
        
        # Кодирование категориальных переменных
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            self.label_encoders[col] = le
            print(f"✅ Закодирован: {col}")
        
        # Feature engineering
        if 'credit_amount' in df_processed.columns and 'age' in df_processed.columns:
            df_processed['amount_to_age_ratio'] = df_processed['credit_amount'] / (df_processed['age'] + 1)
            print("✅ Создан признак: amount_to_age_ratio")
        
        if 'duration' in df_processed.columns and 'credit_amount' in df_processed.columns:
            df_processed['duration_to_amount_ratio'] = df_processed['duration'] / (df_processed['credit_amount'] + 1)
            print("✅ Создан признак: duration_to_amount_ratio")
        
        self.X = df_processed
        self.y = y
        
        print(f"✅ Предобработка завершена. Признаков: {self.X.shape[1]}")
        
        return self.X, self.y
    
    def train_models(self):
        """
        Обучение нескольких ML моделей
        """
        print("\n🤖 ОБУЧЕНИЕ МОДЕЛЕЙ...")
        print("-" * 50)
        
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Масштабирование
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"📊 Обучающая выборка: {X_train.shape[0]} клиентов")
        print(f"📊 Тестовая выборка: {X_test.shape[0]} клиентов")
        
        # Модели для сравнения
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
        }
        
        # Обучение и оценка моделей
        for name, model in self.models.items():
            print(f"🔄 Обучение {name}...")
            
            try:
                # Кросс-валидация
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
                
                # Обучение
                model.fit(X_train_scaled, y_train)
                
                # Предсказания
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Метрики
                accuracy = accuracy_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                self.results[name] = {
                    'model': model,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"   ✅ CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                print(f"   ✅ Test Accuracy: {accuracy:.3f}")
                print(f"   ✅ ROC-AUC: {roc_auc:.3f}")
                
            except Exception as e:
                print(f"   ❌ Ошибка: {e}")
        
        # Выбор лучшей модели
        if self.results:
            self.best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['roc_auc'])
            self.best_model = self.results[self.best_model_name]['model']
            self.best_accuracy = self.results[self.best_model_name]['test_accuracy']
            self.best_auc = self.results[self.best_model_name]['roc_auc']
            
            print(f"\n🎯 ЛУЧШАЯ МОДЕЛЬ: {self.best_model_name}")
            print(f"🏆 ТОЧНОСТЬ: {self.best_accuracy:.3f}")
            print(f"🎯 ROC-AUC: {self.best_auc:.3f}")
        
        self.X_test = X_test
        self.y_test = y_test
        self.X_test_scaled = X_test_scaled
        
        return self.results
    
    def evaluate_models(self):
        """
        Детальная оценка и визуализация результатов
        """
        if not self.results:
            print("❌ Модели не обучены!")
            return
        
        print("\n📈 ОЦЕНКА РЕЗУЛЬТАТОВ...")
        print("-" * 50)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('РЕЗУЛЬТАТЫ МОДЕЛИ КРЕДИТНОГО СКОРИНГА', fontsize=16, fontweight='bold')
        
        # 1. Матрица ошибок лучшей модели
        y_pred_best = self.results[self.best_model_name]['predictions']
        cm = confusion_matrix(self.y_test, y_pred_best)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0],
                xticklabels=['Предсказан Хороший', 'Предсказан Плохой'],
                yticklabels=['Фактический Хороший', 'Фактический Плохой'])
        axes[0,0].set_title('Матрица ошибок', fontweight='bold')
        
        # 2. ROC-кривые
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
            axes[0,1].plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.3f})', linewidth=2)
        
        axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Случайная модель')
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC-кривые', fontweight='bold')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Важность признаков
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=True).tail(10)
            
            axes[1,0].barh(range(len(feature_importance)), feature_importance['importance'])
            axes[1,0].set_yticks(range(len(feature_importance)))
            axes[1,0].set_yticklabels(feature_importance['feature'])
            axes[1,0].set_title('Топ-10 важных признаков', fontweight='bold')
            axes[1,0].set_xlabel('Важность')
        
        # 4. Сравнение моделей
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['test_accuracy'] for name in model_names]
        
        bars = axes[1,1].bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'gold', 'lightcoral'])
        axes[1,1].set_ylabel('Accuracy')
        axes[1,1].set_title('Сравнение точности моделей', fontweight='bold')
        axes[1,1].set_ylim(0, 1)
        
        for bar, acc in zip(bars, accuracies):
            axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_results.png', dpi=300, bbox_inches='tight')
        print("✅ Результаты сохранены в 'model_results.png'")
        plt.show()
    
    def generate_report(self):
        """
        Генерация финального отчета
        """
        print("\n" + "=" * 80)
        print("🏆 ФИНАЛЬНЫЙ ОТЧЕТ ПРОЕКТА")
        print("=" * 80)
        
        print(f"📊 ДАТАСЕТ: German Credit Data (UCI Machine Learning Repository)")
        print(f"📈 РАЗМЕР ДАННЫХ: {self.df.shape[0]} записей, {self.df.shape[1]} признаков")
        print(f"🎯 ЗАДАЧА: Бинарная классификация кредитного риска")
        print(f"🤖 ЛУЧШАЯ МОДЕЛЬ: {self.best_model_name}")
        print(f"📊 ТОЧНОСТЬ: {self.best_accuracy:.3f}")
        print(f"🎯 ROC-AUC: {self.best_auc:.3f}")
        
        if self.best_auc > 0.8:
            quality = "ОТЛИЧНОЕ"
        elif self.best_auc > 0.7:
            quality = "ХОРОШЕЕ"
        else:
            quality = "ТРЕБУЕТ УЛУЧШЕНИЯ"
        
        print(f"✅ КАЧЕСТВО: {quality}")
        
        print(f"\n💼 БИЗНЕС-ЦЕННОСТЬ:")
        print("• Автоматизация оценки кредитных заявок")
        print("• Снижение риска дефолтов")
        print("• Ускорение процесса кредитования")
        
        print(f"\n🚀 СЛЕДУЮЩИЕ ШАГИ:")
        print("1. Настройка порога классификации под риск-аппетит")
        print("2. Добавление мониторинга дрейфа данных")
        print("3. Интеграция с CRM системой")
        
    def create_dashboard(self, port=8050):
        """Создание дашборда"""
        if not hasattr(self, 'results'):
            print("❌ Сначала обучите модели!")
            return
        
        try:
            from dashboard import CreditScoringDashboard
            
            print("\n🎮 СОЗДАНИЕ ПРОФЕССИОНАЛЬНОГО ДАШБОРДА...")
            print("📊 Все графики будут в одном веб-приложении")
            print("🌐 Откроется в одной вкладке браузера")
            
            dashboard = CreditScoringDashboard(
                df=self.df,
                results=self.results,
                X_test=self.X_test,
                y_test=self.y_test,
                feature_names=self.X.columns
            )
            
            dashboard.run(port=port)
            
        except ImportError as e:
            print(f"❌ Не удалось создать дашборд: {e}")
            print("💡 Установите: pip install dash")

def main():
    """
    Основная функция запуска пайплайна
    """
    print("🎯 ЗАПУСК ПРОЕКТА: УМНЫЙ КРЕДИТНЫЙ СКОРИНГ")
    print("=" * 80)
    
    # Инициализация модели
    credit_model = CreditScoringModel()
    
    # Полный ML пайплайн
    credit_model.load_data(source='online')  # или 'local' для локального файла
    credit_model.explore_data()
    credit_model.preprocess_data()
    credit_model.train_models()
    credit_model.evaluate_models()
    credit_model.generate_report()

if __name__ == "__main__":
    main()