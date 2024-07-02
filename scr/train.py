#%%
# Importação das bibliotecas

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline

from sklearn import tree
from sklearn import linear_model
from sklearn import ensemble

from feature_engine import imputation
from feature_engine import encoding
# %%
# Carregamento dos dados

df = pd.read_csv("../data/processed/train.csv")
df.head()
# %%
# Separação dos dados em treino e teste

features = df.columns.to_list()[1:-1]
target = 'AdoptionLikelihood'

X_train, X_test, y_train, y_test = model_selection.train_test_split(df[features],
                                                                    df[target],
                                                                    test_size=0.20,
                                                                    random_state=42,
                                                                    stratify=df[target])

print("Taxa de Reposta Treino: ", y_train.mean())
print("Taxa de Reposta Teste: ", y_test.mean())
# %%
cat_features = X_test.select_dtypes(exclude = 'number').columns.to_list()
num_features = X_test.select_dtypes(include = 'number').columns.to_list()

# Imputações

num_imputer = imputation.MeanMedianImputer(imputation_method = 'median', variables = num_features)
cat_imputer = imputation.CategoricalImputer(imputation_method = 'frequent', variables = cat_features)

# Encoding
onehot = encoding.OneHotEncoder(variables=cat_features)

# Modelo escolhido
model = linear_model.LogisticRegression(max_iter=1000, random_state=42)

# Grid para os melhores hiper parâmetros
params = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'sag']
}

grid = model_selection.GridSearchCV(model, 
                                    param_grid = params,
                                    n_jobs=-1,
                                    cv = 5,
                                    scoring = "roc_auc")

# Pipeline
lr = pipeline.Pipeline([
    ('imp_1', num_imputer),
    ('imp_2', cat_imputer),
    ('ohe', onehot),
    ('model', grid)
])

lr.fit(X_train, y_train)
# %%
# Predições

y_train_predict = lr.predict(X_train)
y_train_proba = lr.predict_proba(X_train)[:,1]

y_test_predict = lr.predict(X_test)
y_test_proba = lr.predict_proba(X_test)[:,1]
# %%
acc_train = metrics.accuracy_score(y_train, y_train_predict)
acc_test = metrics.accuracy_score(y_test, y_test_predict)
print(f"Acurácia da base de treino: {acc_train}")
print(f"Acurácia da base de teste: {acc_test}")

auc_train = metrics.roc_auc_score(y_train, y_train_proba)
auc_test = metrics.roc_auc_score(y_test, y_test_proba)
print(f"AUC da base de treino: {auc_train}")
print(f"AUC da base de teste: {auc_test}")

acc_diff = (acc_train - acc_test) * 100
auc_diff = (auc_train - auc_test) * 100
print(f"Diferença de {round(acc_diff, 4)}% entre treino e teste na Acurácia")
print(f"Diferença de {round(auc_diff, 4)}% entre treino e teste para AUC")
# %%
roc_curve = metrics.roc_curve(y_test, y_test_proba)
plt.plot(roc_curve[0], roc_curve[1] )
plt.grid(True)
plt.plot([0,1], [0,1], '--')
plt.show()
# %%
model_pickle = pd.Series({
    "model": lr,
    "features": features,
    "auc_test": auc_test
})

model_pickle.to_pickle("modelo_pets.pkl")