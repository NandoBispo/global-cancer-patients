# === modelo.py ===
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from pre_processamento import carregar_dados_processados

# Carregar dados
dados = carregar_dados_processados()

# Variável alvo
dados['Faixa_Survival_Binaria'] = dados['Faixa_Survival_Binaria'].map({'Curto Prazo': 0, 'Longo Prazo': 1})

# Seleção de variáveis categóricas para dummificação
variaveis_categoricas = ["Alcohol_Use_Categoria", "Genetic_Risk_Categoria", "Air_Pollution_Categoria",
                         "Obesity_Level_Categoria", "Smoking_Categoria"]
dados_dummy = pd.get_dummies(dados, columns=variaveis_categoricas, drop_first=True)

# Remover colunas irrelevantes e categóricas originais
colunas_remover = ['Faixa_Survival', 'Cancer_Type', 'Country_Region', 'Survival_Years']
X = dados_dummy.drop(columns=colunas_remover + ['Faixa_Survival_Binaria'], errors='ignore')
y = dados_dummy['Faixa_Survival_Binaria']

# Separar em treino/teste (opcional, aqui só treinamos o final)
modelo = BaggingClassifier()
modelo.fit(X, y)

# Salvar modelo
joblib.dump(modelo, 'modelo_bagging.pkl')
joblib.dump(X.columns.tolist(), 'colunas_modelo.pkl')
