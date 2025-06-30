# preprocessamento.py

import pandas as pd
import numpy as np

def carregar_dados_processados():
    url = 'https://raw.githubusercontent.com/NandoBispo/global-cancer-patients/main/dados/global_cancer_patients_2015_2024.csv'
    df = pd.read_csv(url)

    dados = df.drop(columns=['Patient_ID'])

    # Gênero
    dados = dados[dados['Gender'].isin(['Male', 'Female'])]
    dados['Gender'] = np.where(dados['Gender'] == 'Female', 1, 0)

    # Estágio do Câncer
    stage = {
        'Stage 0': 0,
        'Stage I': 1,
        'Stage II': 2,
        'Stage III': 3,
        'Stage IV': 4
    }
    dados['Cancer_Stage_Numeric'] = dados['Cancer_Stage'].map(stage)
    dados = dados.drop(columns=['Cancer_Stage'])

    # Uso de Álcool
    quartis = dados['Alcohol_Use'].quantile([0.25, 0.5, 0.75])
    def categorizar_alcool(v):
        if v <= quartis[0.25]:
            return 'Baixo Consumo'
        elif v <= quartis[0.5]:
            return 'Consumo Moderado'
        elif v <= quartis[0.75]:
            return 'Consumo Alto'
        else:
            return 'Consumo Muito Alto'
    dados['Alcohol_Use_Categoria'] = dados['Alcohol_Use'].apply(categorizar_alcool)
    dados = dados.drop(columns=['Alcohol_Use'])

    # Risco Genético
    quartis = dados['Genetic_Risk'].quantile([0.25, 0.5, 0.75])
    def categorizar_risco(v):
        if v <= quartis[0.25]:
            return 'Baixo'
        elif v <= quartis[0.5]:
            return 'Moderado'
        elif v <= quartis[0.75]:
            return 'Alto'
        else:
            return 'Muito Alto'
    dados['Genetic_Risk_Categoria'] = dados['Genetic_Risk'].apply(categorizar_risco)
    dados = dados.drop(columns=['Genetic_Risk'])

    # Poluição do Ar
    quartis = dados['Air_Pollution'].quantile([0.25, 0.5, 0.75])
    def categorizar_poluicao(v):
        if v <= quartis[0.25]:
            return 'Baixo'
        elif v <= quartis[0.5]:
            return 'Moderado'
        elif v <= quartis[0.75]:
            return 'Alto'
        else:
            return 'Muito Alto'
    dados['Air_Pollution_Categoria'] = dados['Air_Pollution'].apply(categorizar_poluicao)
    dados = dados.drop(columns=['Air_Pollution'])

    # Nível de Obesidade
    quartis = dados['Obesity_Level'].quantile([0.25, 0.5, 0.75])
    def categorizar_obesidade(v):
        if v <= quartis[0.25]:
            return 'Baixo'
        elif v <= quartis[0.5]:
            return 'Moderado'
        elif v <= quartis[0.75]:
            return 'Alto'
        else:
            return 'Muito Alto'
    dados['Obesity_Level_Categoria'] = dados['Obesity_Level'].apply(categorizar_obesidade)
    dados = dados.drop(columns=['Obesity_Level'])

    # Fumante
    quartis = dados['Smoking'].quantile([0.25, 0.5, 0.75])
    def categorizar_fumo(v):
        if v <= quartis[0.25]:
            return 'Baixo'
        elif v <= quartis[0.5]:
            return 'Moderado'
        elif v <= quartis[0.75]:
            return 'Alto'
        else:
            return 'Muito Alto'
    dados['Smoking_Categoria'] = dados['Smoking'].apply(categorizar_fumo)
    dados = dados.drop(columns=['Smoking'])

    # Normalização e custo
    dados['Treatment_Cost_USD'] = dados['Treatment_Cost_USD'] / 100000
    dados['Age'] = (dados['Age'] - np.mean(dados['Age'])) / np.std(dados['Age'])

    # Variável de interesse
    # dados['Survived'] = dados['Survival_Years'].apply(lambda x: 1 if x > 0 else 0)
    def classificar_binaria(faixa):
        if faixa in ['menos de 1 ano', '1 a 3 anos']:
            return 'Curto Prazo'
        else:
            return 'Longo Prazo'

    dados['Faixa_Survival_Binaria'] = dados['Faixa_Survival'].apply(classificar_binaria)

    return dados
