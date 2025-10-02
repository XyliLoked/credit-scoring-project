# dashboard.py
import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

class CreditScoringDashboard:
    """
    Профессиональный дашборд для анализа кредитного скоринга
    Все графики в одном веб-приложении с навигацией по вкладкам
    """
    
    def __init__(self, df, results, X_test=None, y_test=None, feature_names=None):
        self.df = df
        self.results = results
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        
        self.app = dash.Dash(__name__)
        self._setup_styles()
        self._setup_layout()
        self._setup_callbacks()
    
    def _setup_styles(self):
        """Настройка CSS стилей"""
        self.styles = {
            'header': {
                'textAlign': 'center', 
                'color': '#2c3e50', 
                'marginBottom': 30,
                'fontFamily': 'Arial, sans-serif'
            },
            'tab': {
                'padding': '20px',
                'fontFamily': 'Arial, sans-serif'
            }
        }

    def _create_metrics_cards(self):
        """Создание карточек с ключевыми метриками"""
        cards = []
        
        # Карточка 1: Общие данные
        cards.append(html.Div([
            html.H4("📈 Данные", style={'color': '#3498db', 'marginBottom': 10}),
            html.P(f"Записей: {len(self.df)}", style={'margin': 5}),
            html.P(f"Признаков: {len(self.df.columns) - 1}", style={'margin': 5}),
            html.P(f"Целевых классов: {self.df['target'].nunique()}", style={'margin': 5})
        ], style={
            'padding': '20px', 
            'backgroundColor': 'white',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'width': '23%'
        }))
        
        # Карточка 2: Распределение классов
        target_counts = self.df['target'].value_counts()
        good_percent = (target_counts[0] / len(self.df)) * 100 if 0 in target_counts else 0
        cards.append(html.Div([
            html.H4("🎯 Классы", style={'color': '#27ae60', 'marginBottom': 10}),
            html.P(f"Хорошие: {target_counts[0]} ({good_percent:.1f}%)", style={'margin': 5}),
            html.P(f"Плохие: {target_counts[1]} ({100-good_percent:.1f}%)", style={'margin': 5}),
            html.P(f"Дисбаланс: {abs(50-good_percent):.1f}%", style={'margin': 5})
        ], style={
            'padding': '20px', 
            'backgroundColor': 'white',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'width': '23%'
        }))
        
        # Карточка 3: Лучшая модель
        if self.results:
            best_model = max(self.results.items(), key=lambda x: x[1]['roc_auc'])
            cards.append(html.Div([
                html.H4("🏆 Лучшая модель", style={'color': '#e74c3c', 'marginBottom': 10}),
                html.P(f"{best_model[0]}", style={'margin': 5, 'fontWeight': 'bold'}),
                html.P(f"Accuracy: {best_model[1]['test_accuracy']:.3f}", style={'margin': 5}),
                html.P(f"ROC-AUC: {best_model[1]['roc_auc']:.3f}", style={'margin': 5})
            ], style={
                'padding': '20px', 
                'backgroundColor': 'white',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'width': '23%'
            }))
        
        # Карточка 4: Бизнес-ценность
        cards.append(html.Div([
            html.H4("💼 Бизнес-ценность", style={'color': '#9b59b6', 'marginBottom': 10}),
            html.P("Автоматизация скоринга", style={'margin': 5}),
            html.P("Снижение рисков", style={'margin': 5}),
            html.P("Ускорение решений", style={'margin': 5})
        ], style={
            'padding': '20px', 
            'backgroundColor': 'white',
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
            'width': '23%'
        }))
        
        return cards

    def _setup_layout(self):
        """Создание структуры дашборда"""
        self.app.layout = html.Div([
            html.H1("🏦 Дашборд анализа кредитного скоринга", 
                style=self.styles['header']),
            
            html.Div(self._create_metrics_cards(), id='metrics-cards',
                    style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': 30}),
            
            dcc.Tabs(id="main-tabs", value='eda', children=[
                dcc.Tab(label='📊 Анализ данных', value='eda', children=[
                    html.Div([
                        html.H3("Exploratory Data Analysis (EDA)", 
                            style={'color': '#2c3e50', 'marginBottom': 20}),
                        dcc.Graph(id='target-distribution'),
                        dcc.Graph(id='numeric-features-distribution'),
                        dcc.Graph(id='correlation-heatmap')
                    ], style=self.styles['tab'])
                ]),
                
                dcc.Tab(label='🤖 Анализ моделей', value='models', children=[
                    html.Div([
                        html.H3("Сравнение моделей машинного обучения",
                            style={'color': '#2c3e50', 'marginBottom': 20}),
                        dcc.Graph(id='model-comparison-chart'),
                        dcc.Graph(id='roc-curves-chart'),
                        dcc.Graph(id='confusion-matrix-chart')
                    ], style=self.styles['tab'])
                ]),
                
                dcc.Tab(label='🔍 Важность признаков', value='features', children=[
                    html.Div([
                        html.H3("Анализ важности признаков",
                            style={'color': '#2c3e50', 'marginBottom': 20}),
                        dcc.Dropdown(
                            id='model-feature-selector',
                            options=[{'label': name, 'value': name} for name in self.results.keys()],
                            value=list(self.results.keys())[0] if self.results else '',
                            style={'width': '50%', 'marginBottom': 20}
                        ),
                        dcc.Graph(id='feature-importance-chart')
                    ], style=self.styles['tab'])
                ])
            ]),
            
            html.Div(id="status-bar", children=[
                html.P(f"✅ Загружено записей: {len(self.df)} | Моделей: {len(self.results)}",
                    style={'margin': 0, 'color': '#7f8c8d'})
            ], style={
                'position': 'fixed', 'bottom': 0, 'width': '100%', 
                'backgroundColor': '#ecf0f1', 'padding': '10px 20px',
                'borderTop': '2px solid #bdc3c7', 'fontSize': '14px'
            })
        ], style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'})

    def _setup_callbacks(self):
        """Настройка интерактивности"""
        
        @self.app.callback(
            Output('target-distribution', 'figure'),
            Input('main-tabs', 'value')
        )
        def update_target_distribution(selected_tab):
            if selected_tab == 'eda':
                target_counts = self.df['target'].value_counts()
                fig = px.pie(
                    values=target_counts.values, 
                    names=['Хорошие клиенты', 'Плохие клиенты'],
                    title='Распределение классов клиентов',
                    color_discrete_sequence=['#27ae60', '#e74c3c'],
                    hole=0.4
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                return fig
            return go.Figure()
        
        @self.app.callback(
            Output('numeric-features-distribution', 'figure'),
            Input('main-tabs', 'value')
        )
        def update_numeric_features(selected_tab):
            if selected_tab == 'eda':
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                numeric_cols = [col for col in numeric_cols if col != 'target']
                
                if len(numeric_cols) > 0 and 'credit_amount' in self.df.columns:
                    fig = px.histogram(
                        self.df, 
                        x='credit_amount', 
                        color='target',
                        nbins=30, 
                        barmode='overlay',
                        title='Распределение суммы кредита по классам',
                        color_discrete_sequence=['#27ae60', '#e74c3c']
                    )
                    return fig
            return go.Figure()
        
        @self.app.callback(
            Output('correlation-heatmap', 'figure'),
            Input('main-tabs', 'value')
        )
        def update_correlation_heatmap(selected_tab):
            if selected_tab == 'eda':
                numeric_data = self.df.select_dtypes(include=[np.number])
                if len(numeric_data.columns) > 1:
                    corr_matrix = numeric_data.corr()
                    fig = px.imshow(
                        corr_matrix,
                        title='Корреляционная матрица признаков',
                        color_continuous_scale='RdBu_r',
                        aspect="auto"
                    )
                    return fig
            return go.Figure()
        
        @self.app.callback(
            Output('model-comparison-chart', 'figure'),
            Input('main-tabs', 'value')
        )
        def update_model_comparison(selected_tab):
            if selected_tab == 'models' and self.results:
                model_names = list(self.results.keys())
                accuracies = [self.results[name]['test_accuracy'] for name in model_names]
                auc_scores = [self.results[name]['roc_auc'] for name in model_names]
                
                fig = go.Figure(data=[
                    go.Bar(name='Accuracy', x=model_names, y=accuracies,
                        marker_color='#3498db', text=[f'{acc:.3f}' for acc in accuracies],
                        textposition='auto'),
                    go.Bar(name='ROC-AUC', x=model_names, y=auc_scores,
                        marker_color='#9b59b6', text=[f'{auc:.3f}' for auc in auc_scores],
                        textposition='auto')
                ])
                fig.update_layout(
                    title='Сравнение производительности моделей',
                    barmode='group',
                    yaxis_title='Score',
                    yaxis_range=[0, 1]
                )
                return fig
            return go.Figure()
        
        @self.app.callback(
            Output('roc-curves-chart', 'figure'),
            Input('main-tabs', 'value')
        )
        def update_roc_curves(selected_tab):
            if selected_tab == 'models' and self.results and self.X_test is not None:
                fig = go.Figure()
                colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']
                
                for i, (name, result) in enumerate(self.results.items()):
                    if i >= len(colors):
                        break
                    fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
                    auc_score = roc_auc_score(self.y_test, result['probabilities'])
                    
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr, 
                        name=f'{name} (AUC={auc_score:.3f})',
                        line=dict(color=colors[i], width=3)
                    ))
                
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1], 
                    name='Случайная модель',
                    line=dict(dash='dash', color='gray')
                ))
                
                fig.update_layout(
                    title='ROC-кривые моделей',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate'
                )
                return fig
            return go.Figure()

    def run(self, debug=False, port=8050):
        """Запуск дашборда"""
        print(f"\n🚀 ЗАПУСК ДАШБОРДА...")
        print(f"📊 Откройте в браузере: http://localhost:{port}")
        print(f"⏹️  Для остановки нажмите Ctrl+C")
        print("-" * 50)
        
        try:
            self.app.run(debug=debug, port=port)
        except Exception as e:
            print(f"❌ Ошибка запуска дашборда: {e}")