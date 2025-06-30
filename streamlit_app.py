# === app.py ===
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from pre_processamento import carregar_dados_processados

st.set_page_config(page_title="Predição de Sobrevivência", layout="wide")

# Navegação lateral
opcoes = ['Boas-vindas', 'Processo', 'Previsão', 'Dashboard']
pagina = st.sidebar.selectbox('📌 Navegue pelo menu:', opcoes)

# ===========================
# 📌 Página de Boas-vindas
# ===========================
if pagina == 'Boas-vindas':
    st.title("🔬 Predição de Sobrevivência de Pacientes com Câncer")
    st.markdown("""
    Esta aplicação visa prever a **probabilidade de um paciente sobreviver a longo prazo**
    com base em características clínicas e ambientais.  
    Os dados utilizados são **simulados** e representam o período de **2015 a 2024**.
    
    A variável predita é:
    - **Faixa_Survival_Binaria**: 'Curto Prazo' (≤3 anos) ou 'Longo Prazo' (>3 anos).
    """)

# ===========================
# ⚙️ Página de Processo
# ===========================
elif pagina == 'Processo':
    st.title("⚙️ Processo e Justificativas")
    st.markdown("""
    - **Pré-processamento**: Limpeza de dados, categorização e dummificação.
    - **Modelos testados**: Regressão Logística, Árvores, KNN, Random Forest, etc.
    - **Modelo escolhido**: O melhor desempenho foi com **BaggingClassifier**, conforme a métrica F1-score e Acurácia Balanceada.
    """)

# ===========================
# 🔮 Página de Previsão
# ===========================
elif pagina == 'Previsão':
    st.title("🔮 Previsão de Sobrevivência a Longo Prazo")

    try:
        modelo = joblib.load('modelo_bagging.pkl')
        colunas_modelo = joblib.load('colunas_modelo.pkl')
    except FileNotFoundError:
        st.error("❌ Modelo não encontrado. Certifique-se de executar `modelo.py` para treinar e salvar o modelo.")
    else:
        st.markdown("Preencha os dados abaixo para obter a previsão:")

        idade = st.slider("Idade", 0, 100, 50)
        alcool = st.selectbox("Consumo de Álcool", ["Baixo", "Médio", "Alto"])
        genetico = st.selectbox("Risco Genético", ["Baixo", "Médio", "Alto"])
        poluicao = st.selectbox("Nível de Poluição", ["Baixo", "Médio", "Alto"])
        obesidade = st.selectbox("Nível de Obesidade", ["Baixo", "Médio", "Alto"])
        fumo = st.selectbox("Nível de Tabagismo", ["Baixo", "Médio", "Alto"])

        if st.button("🔍 Prever"):
            entrada = {
                'Age': idade,
                f"Alcohol_Use_Categoria_{alcool}": 1,
                f"Genetic_Risk_Categoria_{genetico}": 1,
                f"Air_Pollution_Categoria_{poluicao}": 1,
                f"Obesity_Level_Categoria_{obesidade}": 1,
                f"Smoking_Categoria_{fumo}": 1
            }

            X_novo = pd.DataFrame(columns=colunas_modelo)
            X_novo.loc[0] = 0  # Preenche com zeros
            for col, val in entrada.items():
                if col in X_novo.columns:
                    X_novo.at[0, col] = val

            pred = modelo.predict(X_novo)[0]
            prob = modelo.predict_proba(X_novo)[0][pred]

            st.success(f"🧬 Sobrevivência Prevista: {'Longo Prazo' if pred == 1 else 'Curto Prazo'}")
            st.write(f"📊 Probabilidade: {prob:.2%}")

# ===========================
# 📊 Página de Dashboard
# ===========================
elif pagina == 'Dashboard':
    st.title("📊 Dashboard - Dados Tratados")

    dados = carregar_dados_processados()

    col1, col2, col3 = st.columns(3)

    regiao = col1.selectbox("Região", dados['Country_Region'].unique())
    sexo = col2.selectbox("Sexo", ['Masculino', 'Feminino'])
    ano = col3.selectbox("Ano", sorted(dados['Year'].unique()))

    sexo = 'Male' if sexo == 'Masculino' else 'Female'

    filtro = (
        (dados['Country_Region'] == regiao) &
        (dados['Gender'] == sexo) &
        (dados['Year'] == ano)
    )
    dados_filtrados = dados[filtro]

    col1, col2 = st.columns([1, 3])

    col1.metric('Idade Média', round(dados_filtrados['Age'].mean(), 1))
    col1.metric('Tempo Médio de Vida', round(dados_filtrados['Survival_Years'].mean(), 1))
    col1.metric('Custo Médio do Tratamento', round(dados_filtrados['Treatment_Cost_USD'].mean(), 1))
    col1.metric('Estágio III do Câncer',
                '{:.2%}'.format(dados_filtrados['Cancer_Stage'].value_counts(normalize=True).get('Stage III', 0)))
    col1.metric('Estágio IV do Câncer',
                '{:.2%}'.format(dados_filtrados['Cancer_Stage'].value_counts(normalize=True).get('Stage IV', 0)))

    fig = sns.scatterplot(data=dados_filtrados, x='Survival_Years', y='Treatment_Cost_USD', hue='Cancer_Stage')
    plt.xlabel('Anos de Sobrevivência')
    plt.ylabel('Custo do Tratamento (USD)')
    plt.title('Anos de Sobrevivência X Custo')

    col2.pyplot(fig.get_figure())

    st.markdown("---")
