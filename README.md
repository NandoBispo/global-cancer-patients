# 🎈 Blank app template
# 🔬 Predição de Prognóstico de Pacientes com Câncer

Este projeto apresenta uma aplicação interativa construída com **Streamlit** para prever a **gravidade do prognóstico de pacientes com câncer** com base em características clínicas, ambientais e genéticas.

![Banner](https://img.shields.io/badge/streamlit-cloud-blue?style=flat&logo=streamlit)

---

## 🚀 Acesse o App

👉 [Clique aqui para acessar a aplicação no Streamlit Cloud](https://NOME_DO_SEU_APP.streamlit.app)

> A aplicação é totalmente interativa e pode ser utilizada diretamente no navegador.

---

## 📌 Objetivo

O objetivo do projeto é utilizar **técnicas de Machine Learning supervisionado** para prever se um paciente possui **prognóstico de Alta ou Baixa Gravidade**, com base em dados relacionados a:

- Idade, gênero e país de origem
- Estágio do câncer
- Fatores de risco (tabagismo, poluição, obesidade, etc.)
- Risco genético e custo do tratamento
- Tipo de câncer e ano do diagnóstico

---

## 🧠 Metodologia

- **Engenharia da variável-alvo:**  
  A variável `Prognostico` (Alta Gravidade / Baixa Gravidade) foi criada a partir do `Target_Severity_Score`, com ponto de corte na mediana.

- **Pré-processamento:**
  - Normalização com `StandardScaler` para variáveis numéricas
  - Codificação com `OneHotEncoder` para variáveis categóricas
  - Conversão ordinal da variável `Cancer_Stage`

- **Modelagem:**
  - Avaliação de 5 modelos: Regressão Logística, Árvore de Decisão, KNN, Bagging e Random Forest
  - Validação cruzada estratificada (10 folds)
  - Métrica principal: F1-Score (com correção para divisão por zero)

- **Melhor modelo selecionado:**  
  O modelo com melhor desempenho foi exportado em formato `.pkl` e utilizado na aplicação via `Pipeline`.

---

## 📊 Funcionalidades

A aplicação possui as seguintes páginas:

- **Boas-vindas:** introdução ao projeto
- **Previsão:** formulário interativo que permite o preenchimento de dados para realizar a predição de gravidade
- **Dashboard:** gráficos e estatísticas interativas por país, sexo, ano e tipo de câncer
- **Sobre o Projeto:** detalhes completos sobre os dados, abordagem de modelagem e bibliotecas utilizadas

---

## 🧾 Fonte dos Dados

Os dados utilizados neste projeto são **sintéticos** e foram obtidos na plataforma Kaggle:

🔗 [Global Cancer Patients 2015–2024 – Kaggle Dataset](https://www.kaggle.com/datasets/zahidmughal2343/global-cancer-patients-2015-2024)

---

## 🛠️ Tecnologias Utilizadas

- `Python 3.11`
- `Streamlit`
- `pandas`, `numpy`
- `scikit-learn 1.7.0`
- `matplotlib`, `seaborn`
- `joblib`, `pickle`
- `ydata-profiling`
- `Streamlit Cloud + GitHub`

---

## 👨‍💻 Como Executar Localmente

1. Clone este repositório:
   ```bash
   git clone https://github.com/NandoBispo/global-cancer-patients.git
   cd global-cancer-patients

