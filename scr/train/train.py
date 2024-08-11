#%% Importação das bibliotecas

# Manipulação de dados
import pandas as pd
import numpy as np

# Análise Exploratória de Dados
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from lightgbm import LGBMClassifier

# Pré-processamento
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from feature_engine.imputation import CategoricalImputer, MeanMedianImputer
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
# %% Carregamento dos dados

df = pd.read_csv("../../data/raw/pet_adoption_data.csv")
df.head()
# %% Separação dos dados em treino e teste
features = df.drop(columns = ['PetID', 'AdoptionLikelihood'], axis = 1).columns.to_list()
target = 'AdoptionLikelihood'

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size = 0.20, stratify=df[target], random_state=42)
# %% Pré-processamento
cat_features = X_test.select_dtypes(exclude = 'number').columns.to_list()
num_features = X_test.select_dtypes(include = 'number').columns.to_list()

cat_transformer = Pipeline([
    ('imput', CategoricalImputer(imputation_method='frequent')),
    ('encoder', TargetEncoder())
])

num_transformer = Pipeline([
    ('imput', MeanMedianImputer(imputation_method='median'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', cat_transformer, cat_features),
        ('num', num_transformer, num_features)
    ]
)
# %% Pipeline do modelo
model = LGBMClassifier(learning_rate=0.05, n_estimators=100, num_leaves=31)

clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

clf.fit(X_train, y_train)
# %% Predições e métricas
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:,1]

def metric_report(y_true, y_proba, threshold = 0.5):
    y_pred = (y_proba > threshold).astype(int)

    recall = metrics.recall_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    f1score = metrics.f1_score(y_true, y_pred)
    accuracy = metrics.accuracy_score(y_true, y_pred)
    rocauc = metrics.roc_auc_score(y_true, y_proba)

    return {
        'Recall': recall,
        'Precision': precision,
        'F1 Score': f1score,
        'ROC AUC': rocauc,
        'Accuracy': accuracy
    }

metricas = metric_report(y_test, y_proba)
metricas
# %%
model_series = pd.Series({
    "model": clf,
    "features": features,
    "auc_test": metricas
})

model_series.to_pickle("../../models/modelo_preditivo.pkl")