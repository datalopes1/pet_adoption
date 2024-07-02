#%%
import pandas as pd
model = pd.read_pickle("modelo_pets.pkl")
model
# %%
df = pd.read_csv("../data/processed/test.csv")
df.shape
# %%
X = df[model['features']]
predict_proba = model['model'].predict_proba(X)[:,1]
predict_proba
# %%
df['ProbAdoption'] = predict_proba
df.head()
# %%
df.to_csv('../data/processed/data_probAdoption.csv', index = False)