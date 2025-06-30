# === app.py ===
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import pickle
from matplotlib.ticker import FuncFormatter
from pre_processamento import carregar_dados_processados

st.set_page_config(page_title="Predição de Sobrevivência", layout="wide")

# Navegação lateral
opcoes = ['Boas-vindas', 'Sobre o Projeto', 'Processo', 'Previsão', 'Dashboard']

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
# 📌 Página Sobre o Projeto
# ===========================

elif pagina == 'Sobre o Projeto':
    st.title("📚 Sobre o Projeto")

    st.markdown("""
    ### 📌 Contexto e Objetivos
    Este projeto tem como objetivo prever a **chance de sobrevivência de pacientes com câncer** com base em características clínicas, genéticas e ambientais.

    O foco é identificar, com o auxílio de modelos de aprendizado de máquina, **pacientes com maior ou menor probabilidade de sobreviver a longo prazo**.

    ### 🗂️ Origem dos Dados
    Os dados utilizados são **simulados**, representando pacientes de diversos países no período de **2015 a 2024**.

    As variáveis incluem:
    - Fatores clínicos: idade, estágio do câncer, anos de sobrevivência, custo do tratamento.
    - Fatores de risco: poluição do ar, tabagismo, obesidade, álcool, risco genético.
    - Indicador de gravidade: `Target_Severity_Score`.

    ### 🛠️ Pré-processamento dos Dados
    O tratamento incluiu:
    - Conversão de variáveis categóricas (como `Cancer_Stage`) em forma numérica.
    - Normalização de variáveis contínuas para melhorar o desempenho dos algoritmos.
    - Criação da variável-alvo binária `Faixa_Survival_Binaria` (Longo Prazo / Curto Prazo).
    - Divisão dos dados em treino/teste e validação cruzada.

    ### 🧰 Pacotes Utilizados
    - **streamlit**: construção da interface interativa.
    - **pandas**: manipulação de dados tabulares.
    - **numpy**: suporte a operações matemáticas e vetorização.
    - **matplotlib & seaborn**: criação de visualizações e gráficos.
    - **scikit-learn**: construção e avaliação dos modelos de machine learning.
    - **joblib**: salvamento e carregamento do modelo treinado (.pkl).
    - **requests**: download de arquivos hospedados remotamente (como o modelo).

    ---
    Este app foi desenvolvido como um exercício de aplicação prática de **Machine Learning interpretável e acessível**.
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
    # st.set_page_config(
    #     page_title="Previsão de Prognóstico de Câncer",
    #     page_icon="🔮",
    #     layout="wide"
    # )

    st.title("🔮 Previsão de Prognóstico de Câncer")
    st.markdown("Preencha os dados do paciente abaixo para obter uma previsão sobre a gravidade do prognóstico.")

    # --- Carregamento do Modelo ---
    @st.cache_resource
    def carregar_modelo():
        try:
            with open('modelo_cancer.pkl', 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
            

    modelo = carregar_modelo()

    if modelo is None:
        st.error("❌ Arquivo do modelo não encontrado! Certifique-se de que `modelo_cancer.pkl` está no mesmo diretório do app.")
    else:
        # --- Entrada de Dados ---
        st.divider()
        st.subheader("Por favor, insira os dados do paciente:")

        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Idade do Paciente", 20, 90, 55)
            gender = st.selectbox("Gênero", ["Male", "Female"])
            treatment_cost = st.number_input("Custo Estimado do Tratamento (USD)", min_value=1000, max_value=200000, value=50000, step=1000)
            cancer_stage_text = st.select_slider(
                "Estágio do Câncer",
                options=['Stage 0', 'Stage I', 'Stage II', 'Stage III', 'Stage IV'],
                value='Stage II'
            )

        with col2:
            genetic_risk = st.slider("Risco Genético (0-10)", 0, 10, 5)
            air_pollution = st.slider("Exposição à Poluição do Ar (0-10)", 0, 10, 5)
            alcohol_use = st.slider("Consumo de Álcool (0-10)", 0, 10, 5)
            smoking = st.slider("Nível de Tabagismo (0-10)", 0, 10, 5)
            obesity_level = st.slider("Nível de Obesidade (0-10)", 0, 10, 5)

        st.divider()

        if st.button("🔍 Realizar Previsão", type="primary", use_container_width=True):
            stage_map = {'Stage 0': 0, 'Stage I': 1, 'Stage II': 2, 'Stage III': 3, 'Stage IV': 4}
            cancer_stage_ordinal = stage_map[cancer_stage_text]

            input_data = pd.DataFrame({
                'Age': [age],
                'Treatment_Cost_USD': [treatment_cost],
                'Genetic_Risk': [genetic_risk],
                'Air_Pollution': [air_pollution],
                'Alcohol_Use': [alcohol_use],
                'Smoking': [smoking],
                'Obesity_Level': [obesity_level],
                'Cancer_Stage_Ordinal': [cancer_stage_ordinal],
                'Gender': [gender]
            })

            st.write("⚙️ **Dados de entrada para o modelo:**")
            st.dataframe(input_data)

            # Realiza previsão
            predicao = modelo.predict(input_data)[0]
            probabilidades = modelo.predict_proba(input_data)

            prob_df = pd.DataFrame(probabilidades, columns=modelo.classes_, index=["Probabilidade"])

            # Resultado
            st.write("---")
            st.subheader("📈 Resultado da Previsão")

            if predicao == 'Alta Gravidade':
                st.error(f"**Prognóstico Previsto:** {predicao}")
            else:
                st.success(f"**Prognóstico Previsto:** {predicao}")

            st.write("O gráfico abaixo mostra a confiança do modelo em cada classe:")
            st.bar_chart(prob_df.T)

            st.info("""
            **Aviso Importante:** O modelo foi treinado com dados sintéticos e apresenta desempenho elevado.
            Os resultados são ilustrativos e não devem ser usados para decisões clínicas.
            """)


# ===========================
# 📊 Página de Dashboard
# ===========================
elif pagina == 'Dashboard':
    st.title("📊 Dashboard - Dados Tratados")

    url = 'https://raw.githubusercontent.com/NandoBispo/global-cancer-patients/main/dados/global_cancer_patients_2015_2024.csv'
    df = pd.read_csv(url)

    col1, col2, col3 = st.columns(3)

    st.markdown('---')

    regiao = col1.selectbox("Região", df['Country_Region'].unique())
    sexo = col2.selectbox("Sexo", ['Masculino', 'Feminino'])
    ano = col3.selectbox("Ano", list(range(2015, 2025)))

    sexo = 'Male' if sexo == 'Masculino' else 'Female'

    filtro_regiao = df['Country_Region'] == regiao
    filtro_sexo = df['Gender'] == sexo
    filtro_ano = df['Year'] == ano

    dados_filtrados = df.loc[filtro_regiao & filtro_sexo & filtro_ano]

    col1, col2 = st.columns([1, 3])

    col1.metric('Idade Média', round(dados_filtrados['Age'].mean(), 1))
    col1.metric('Tempo Médio de Vida (em Anos)', round(dados_filtrados['Survival_Years'].mean(), 1))
    col1.metric('Custo Médio do Tratamento (USD)',
                f"${round(dados_filtrados['Treatment_Cost_USD'].mean(), 2):,.2f}")


    # 🔍 Distribuição dos Estágios do Câncer
    # st.subheader("📊 Distribuição dos Estágios do Câncer")

    # if not dados_filtrados.empty:
    #     distribuicao_estagios = (
    #         dados_filtrados['Cancer_Stage']
    #         .value_counts(normalize=True)
    #         .sort_index()
    #     )

    #     for estagio, proporcao in distribuicao_estagios.items():
    #         st.write(f"- **{estagio}**: {proporcao:.2%}")
    # else:
    #     st.warning("Nenhum dado disponível para os filtros selecionados.")

    if not dados_filtrados.empty:
        distribuicao_estagios = (
            dados_filtrados['Cancer_Stage']
            .value_counts(normalize=True)
            .sort_index()
        )

        col1.markdown("#### 📊 Estágios do Câncer")
        for estagio, proporcao in distribuicao_estagios.items():
            col1.markdown(f"- **{estagio}**: {proporcao:.2%}")
    else:
        col1.warning("Nenhum dado disponível para os filtros selecionados.")

    # 🎯 Gráfico de dispersão: Anos de Sobrevivência vs Custo
    fig = sns.scatterplot(data=dados_filtrados,
                          x='Survival_Years',
                          y='Treatment_Cost_USD')
                        #   hue='Cancer_Stage')
    plt.xlabel('Anos de Sobrevivência')
    plt.ylabel('Custo do Tratamento (USD)')
    plt.title('Anos de Sobrevivência X Custo do Tratamento')
    plt.ticklabel_format(style='plain', axis='y')  # evita notação científica
    plt.gca().get_yaxis().set_major_formatter(
    plt.matplotlib.ticker.FuncFormatter(lambda x, _: f'{int(x):,}'.replace(",", ".")))  # separador de milhar

    col2.pyplot(fig.get_figure())

    st.markdown('---')

