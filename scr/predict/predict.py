#%%
import pandas as pd
model = pd.read_pickle("../../models/modelo_preditivo.pkl")
model
# %%
df = pd.read_csv("../../data/raw/pet_adoption_data.csv")
df.shape
# %%
X = df[model['features']]
predict_proba = model['model'].predict_proba(X)[:,1]
predict_proba
# %%
df['modelPrediction'] = (predict_proba > 0.65).astype(int)
df['probAdoption'] = predict_proba
df.head()
# %%
df.to_excel('../../data/processed/modelPredictions.xlsx', index = False)