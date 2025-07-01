# === app.py ===
import streamlit as st
# import sklearn
# st.write(f"Versão do scikit-learn no ambiente Streamlit: {sklearn.__version__}")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# import joblib
import pickle
from matplotlib.ticker import FuncFormatter
from pre_processamento import carregar_dados_processados

import numpy as np  # <- Essencial para corrigir o erro que você teve

st.set_page_config(page_title="Predição de Sobrevivência", layout="wide")


# Navegação lateral
opcoes = ['Boas-vindas', 'Sobre o Projeto', 'Previsão', 'Dashboard']

pagina = st.sidebar.selectbox('📌 Navegue pelo menu:', opcoes)

# ===========================
# 📌 Página de Boas-vindas
# ===========================
if pagina == 'Boas-vindas':
    st.title("🔬 Predição de Gravidade do Prognóstico para Pacientes com Câncer")
    
    st.markdown("""
    Bem-vindo(a) a esta aplicação interativa que utiliza técnicas de **Machine Learning** para prever o **nível de gravidade do prognóstico** em pacientes diagnosticados com câncer.  
    Aqui, você pode explorar como fatores clínicos, ambientais e demográficos impactam na severidade da doença.

    ---
    ### Sobre o Modelo
    O modelo foi treinado em dados simulados abrangendo pacientes de diversos países, com registros do período entre **2015 e 2024**.  
    A variável alvo predita é o prognóstico categórico:
    - **Alta Gravidade:** pacientes com maior risco e severidade da doença.
    - **Baixa Gravidade:** pacientes com menor risco, prognóstico mais favorável.

    ---
    ### Quais dados são usados para a previsão?
    O modelo considera informações como:
    - Idade e gênero do paciente
    - Tipo e estágio do câncer
    - Fatores de risco: tabagismo, consumo de álcool, obesidade, exposição à poluição do ar, risco genético, entre outros
    - Custo estimado do tratamento

    ---
    ### Para que serve esta aplicação?
    - Demonstrar o potencial do aprendizado de máquina na área da saúde
    - Auxiliar na compreensão dos fatores que influenciam a gravidade do câncer
    - Servir como base para desenvolvimento e experimentação em projetos educacionais e de pesquisa

    ---
    ### Importante
    Este modelo é construído com **dados simulados** e serve para fins educativos e ilustrativos.  
    Não deve ser utilizado para decisões médicas ou clínicas reais.

    ---
    Aproveite para navegar pelas abas e conhecer melhor o projeto, explorar os dados, realizar previsões e entender a metodologia aplicada!
    """)

# ===========================
# 📌 Página Sobre o Projeto
# ===========================

elif pagina == 'Sobre o Projeto':
    st.title("📚 Sobre o Projeto")

    st.markdown("""
    ### 📌 Contexto e Objetivos
    O câncer é uma doença complexa que representa um grande desafio para a saúde pública global. Este projeto visa desenvolver um modelo preditivo de aprendizado de máquina capaz de **estimular a previsão do prognóstico de pacientes com câncer**, classificando-os em "Alta Gravidade" ou "Baixa Gravidade" com base em uma combinação de características clínicas, demográficas e ambientais.

    O objetivo é fornecer uma ferramenta que auxilie na identificação precoce de pacientes com maior risco, para suportar decisões clínicas e estratégias de tratamento, com a ressalva de que o modelo é treinado em dados simulados e serve principalmente para fins educacionais.

    ### 🗂️ Origem e Descrição dos Dados
    Os dados utilizados foram obtidos a partir do repositório público no Kaggle:
    [Global Cancer Patients 2015-2024](https://www.kaggle.com/datasets/zahidmughal2343/global-cancer-patients-2015-2024)

    Este conjunto contém informações simuladas de pacientes diagnosticados com diferentes tipos de câncer entre os anos de 2015 e 2024, abrangendo diversas regiões do mundo.

    As variáveis principais incluem:

    - **Idade:** idade do paciente, variando entre 20 e 90 anos.
    - **Gênero:** masculino, feminino ou outro.
    - **País/Região:** local de origem do paciente.
    - **Tipo de Câncer:** diferentes categorias de câncer (ex.: mama, pulmão, cólon).
    - **Estágio do Câncer:** classificação do estágio da doença, do 0 ao IV.
    - **Fatores de Risco:** níveis de risco genético, exposição à poluição do ar, consumo de álcool, tabagismo, obesidade, entre outros.
    - **Custo do Tratamento:** estimativa do custo em dólares para o tratamento do câncer.
    - **Anos de Sobrevivência:** tempo em anos desde o diagnóstico.
    - **Pontuação de Gravidade (`Target_Severity_Score`):** um escore composto que sintetiza a severidade do câncer para cada paciente.

    ### 🛠️ Pré-processamento dos Dados e Engenharia da Variável-Alvo
    Para viabilizar a modelagem, foi realizada a seguinte preparação dos dados:

    - **Engenharia da variável alvo:** a variável contínua `Target_Severity_Score` foi transformada em uma variável categórica binária chamada `Prognostico`, dividindo os pacientes em duas classes:
      - **Alta Gravidade:** pacientes com escore acima da mediana.
      - **Baixa Gravidade:** pacientes com escore igual ou abaixo da mediana.

    - **Mapeamento do Estágio do Câncer:** os estágios textuais (`Stage 0` a `Stage IV`) foram convertidos em valores ordinais de 0 a 4 para facilitar o processamento.

    - **Remoção de variáveis redundantes ou que gerariam vazamento de dados:** variáveis como `Target_Severity_Score`, `Survival_Years`, `Prognostico` e `Cancer_Stage` foram excluídas das features preditoras.

    - **Tratamento das variáveis categóricas e numéricas:** 
      - Variáveis numéricas foram padronizadas usando `StandardScaler`.
      - Variáveis categóricas foram codificadas via `OneHotEncoder` com tratamento para valores desconhecidos em dados novos.

    - **Divisão e validação:** os dados foram utilizados com validação cruzada estratificada para garantir a robustez das métricas e evitar overfitting.

    ### 🤖 Metodologia e Seleção de Modelos
    Diversos algoritmos de classificação foram avaliados com o objetivo de encontrar o modelo que melhor balanceasse desempenho e interpretabilidade:

    - Regressão Logística
    - Árvore de Decisão
    - K-Nearest Neighbors (KNN)
    - Bagging Classifier
    - Random Forest

    Os modelos foram avaliados por métricas importantes como F1-Score (com foco na classe "Alta Gravidade"), Acurácia Balanceada, Precisão, Revocação e AUC ROC.

    O modelo final escolhido foi o que apresentou o melhor desempenho no F1-Score, garantindo um bom equilíbrio entre precisão e sensibilidade para detectar pacientes de alto risco.

    ### 🧰 Pacotes e Ferramentas Utilizadas
    Para a construção deste projeto foram usados:

    - **streamlit:** desenvolvimento da interface web interativa.
    - **pandas & numpy:** manipulação e análise dos dados.
    - **scikit-learn:** pré-processamento, construção e validação dos modelos de machine learning.
    - **joblib & pickle:** serialização e carregamento do modelo final.
    - **matplotlib & seaborn:** visualizações exploratórias e gráficos.
    - **kagglehub:** download automatizado dos dados do Kaggle.
    - **ydata-profiling:** geração de relatórios exploratórios dos dados.

    ---

    Este projeto é uma demonstração prática do uso de aprendizado de máquina para análise preditiva na área da saúde, com foco em cancerologia. Ressaltamos que, apesar do rigor técnico, o modelo aqui apresentado não deve ser utilizado para decisões clínicas reais, dada a natureza simulada dos dados e o caráter educativo da aplicação.

    """)


elif pagina == 'Previsão':

    @st.cache_data
    def carregar_valores_unicos():
        url = 'https://raw.githubusercontent.com/NandoBispo/global-cancer-patients/main/dados/global_cancer_patients_2015_2024.csv'
        df_dados = pd.read_csv(url)

        traducao_cancer = {
            "Breast": "Mama",
            "Lung": "Pulmão",
            "Colon": "Cólon",
            "Prostate": "Próstata",
            "Liver": "Fígado",
            "Stomach": "Estômago",
            "Cervical": "Colo do útero",
            "Pancreatic": "Pâncreas",
            "Ovarian": "Ovário",
            "Esophageal": "Esôfago"
        }

        df_dados['Cancer_Type_PT'] = df_dados['Cancer_Type'].map(traducao_cancer)

        return {
            'countries': sorted(df_dados['Country_Region'].dropna().unique().tolist()),
            'cancers_en': sorted(df_dados['Cancer_Type'].dropna().unique().tolist()),
            'cancers_pt': sorted(df_dados['Cancer_Type_PT'].dropna().unique().tolist()),
            'anos': sorted(df_dados['Year'].dropna().unique().astype(int).tolist()),
            'mapa_cancer': dict(zip(df_dados['Cancer_Type_PT'], df_dados['Cancer_Type']))
        }

    valores = carregar_valores_unicos()

    st.title("🔮 Previsão de Prognóstico de Câncer")
    st.markdown("Preencha os dados do paciente abaixo para obter uma previsão sobre a gravidade do prognóstico.")

    @st.cache_resource
    def carregar_modelo():
        try:
            with open("modelo_cancer.pkl", "rb") as file:
                return pickle.load(file)
        except Exception as e:
            st.error(f"❌ Erro ao carregar o modelo: {e}")
            return None

    modelo = carregar_modelo()

    if modelo is None:
        st.error("❌ Arquivo do modelo não encontrado! Certifique-se de que `modelo_cancer.pkl` está no mesmo diretório do app.")
    else:
        st.divider()
        st.subheader("📝 Dados do Paciente")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider("Idade do Paciente", 20, 90, 55)
            gender_pt = st.selectbox("Gênero", ["Masculino", "Feminino"])
            gender = 'Male' if gender_pt == 'Masculino' else 'Female'
            country_region = st.selectbox("País/Região", valores['countries'])

        with col2:
            treatment_cost = st.number_input("Custo Estimado do Tratamento (USD)", min_value=1000, max_value=200000, value=50000, step=1000, format="%d")

            cancer_type_pt = st.selectbox("Tipo de Câncer", valores['cancers_pt'])
            cancer_type = valores['mapa_cancer'][cancer_type_pt]

            year = st.selectbox("Ano do Diagnóstico", valores['anos'])

        with col3:
            cancer_stage_text = st.select_slider(
                "Estágio do Câncer",
                options=['Stage 0', 'Stage I', 'Stage II', 'Stage III', 'Stage IV'],
                value='Stage II'
            )
            genetic_risk = st.slider("Risco Genético (0-10)", 0, 10, 5)
            air_pollution = st.slider("Poluição do Ar (0-10)", 0, 10, 5)
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
                'Gender': [gender],
                'Country_Region': [country_region],
                'Cancer_Type': [cancer_type],
                'Year': [year]
            })

            st.write("⚙️ **Visualização dos dados de entrada:**")
            input_display = input_data.copy()
            input_display['Treatment_Cost_USD'] = input_display['Treatment_Cost_USD'].apply(lambda x: f"${x:,.0f}".replace(",", "."))
            input_display['Gender'] = gender_pt
            input_display['Cancer_Type'] = cancer_type_pt
            st.dataframe(input_display)

            try:
                predicao = modelo.predict(input_data)[0]
                probabilidades = modelo.predict_proba(input_data)
                prob_df = pd.DataFrame(probabilidades, columns=modelo.classes_, index=["Probabilidade"])

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
            except Exception as e:
                st.error(f"❌ Erro durante a previsão: {e}")




# ===========================
# 📊 Página de Dashboard
# ===========================
elif pagina == 'Dashboard':
    st.title("📊 Dashboard Interativo - Análise dos Pacientes")

    # Carregamento dos dados
    url = 'https://raw.githubusercontent.com/NandoBispo/global-cancer-patients/main/dados/global_cancer_patients_2015_2024.csv'
    df = pd.read_csv(url)

       # Dicionário de tradução para os tipos de câncer
    traducao_cancer = {
        "Lung": "Pulmão",
        "Breast": "Mama",
        "Colon": "Cólon",
        "Prostate": "Próstata",
        "Stomach": "Estômago",
        "Liver": "Fígado",
        "Pancreatic": "Pâncreas",
        "Leukemia": "Leucemia",
        "Lymphoma": "Linfoma",
        "Skin": "Pele",
        "Other": "Outro"
    }

    tipos_originais = sorted(df['Cancer_Type'].dropna().unique())
    tipos_traduzidos = [traducao_cancer.get(c, c) for c in tipos_originais]

    # ========= Filtros =========
    st.markdown("### 🔍 Filtros Interativos")
    col1, col2, col3, col4 = st.columns(4)

    regiao = col1.selectbox("🌍 Região", sorted(df['Country_Region'].dropna().unique()))
    sexo = col2.selectbox("👤 Sexo", ['Masculino', 'Feminino'])
    ano = col3.selectbox("📅 Ano", sorted(df['Year'].dropna().unique()))
    # tipo_cancer = col4.selectbox("🧬 Tipo de Câncer", sorted(df['Cancer_Type'].dropna().unique()))
    tipo_cancer_traduzido = col4.selectbox("🧬 Tipo de Câncer", tipos_traduzidos)

    sexo = 'Male' if sexo == 'Masculino' else 'Female'
    tipo_cancer = [k for k, v in traducao_cancer.items() if v == tipo_cancer_traduzido]
    tipo_cancer = tipo_cancer[0] if tipo_cancer else tipo_cancer_traduzido

    # ========= Aplicar Filtros =========
    dados_filtrados = df[
        (df['Country_Region'] == regiao) &
        (df['Gender'] == sexo) &
        (df['Year'] == ano) &
        (df['Cancer_Type'] == tipo_cancer)
    ]

    if dados_filtrados.empty:
        st.warning("⚠️ Nenhum dado disponível para os filtros selecionados.")
    else:
        # ========= Métricas =========
        st.markdown("### 📈 Métricas Resumidas")
        col1, col2, col3 = st.columns(3)
        col1.metric('🧓 Idade Média', round(dados_filtrados['Age'].mean(), 1))
        col2.metric('💵 Custo Médio (USD)', f"${round(dados_filtrados['Treatment_Cost_USD'].mean(), 2):,}".replace(",", "."))
        col3.metric('❤️ Sobrevivência Média (anos)', round(dados_filtrados['Survival_Years'].mean(), 1))

        st.divider()

        # ========= Estágios do Câncer =========
        with st.expander("📊 Distribuição dos Estágios do Câncer"):
            ordem_correta = ['Stage 0', 'Stage I', 'Stage II', 'Stage III', 'Stage IV']
            distribuicao_estagios = (
                dados_filtrados['Cancer_Stage']
                .value_counts(normalize=True)
                .reindex(ordem_correta)
                .dropna()
            )
            for estagio, proporcao in distribuicao_estagios.items():
                st.markdown(f"- **{estagio}**: {proporcao:.2%}")

        # ========= Distribuição de Idade =========
        with st.expander("📈 Distribuição da Idade"):
            fig1 = plt.figure(figsize=(6, 4))
            ax1 = sns.histplot(dados_filtrados['Age'], bins=15, kde=True, color="teal")
            ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'.replace(",", ".")))
            plt.title("Distribuição de Idade dos Pacientes")
            plt.xlabel("Idade")
            plt.ylabel("Frequência")
            st.pyplot(fig1)

        # ========= Boxplot do Custo =========
        with st.expander("💵 Custo por Estágio do Câncer"):
            fig2 = plt.figure(figsize=(6, 4))
            ax2 = sns.boxplot(data=dados_filtrados, x="Cancer_Stage", y="Treatment_Cost_USD", order=ordem_correta, palette="Set2")
            ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'.replace(",", ".")))
            plt.title("Custo do Tratamento por Estágio")
            plt.xlabel("Estágio do Câncer")
            plt.ylabel("Custo (USD)")
            st.pyplot(fig2)

        # ========= Dispersão Sobrevivência x Custo =========
        with st.expander("📉 Relação entre Sobrevivência e Custo"):
            fig3 = plt.figure(figsize=(6, 4))
            ax3 = sns.scatterplot(data=dados_filtrados, x='Survival_Years', y='Treatment_Cost_USD', hue='Cancer_Stage', palette='muted')
            ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'.replace(",", ".")))
            plt.title('Anos de Sobrevivência X Custo do Tratamento')
            plt.xlabel("Anos de Sobrevivência")
            plt.ylabel("Custo (USD)")
            st.pyplot(fig3)

        # ========= Heatmap de Correlação =========
        with st.expander("🔬 Mapa de Correlação entre Variáveis Numéricas"):
            numeric_cols = dados_filtrados.select_dtypes(include=np.number).drop(columns=["Year"], errors="ignore")
            corr = numeric_cols.corr()
            fig4 = plt.figure(figsize=(6, 4))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlação entre Variáveis Numéricas")
            st.pyplot(fig4)

        st.markdown("---")
        st.success("🔎 Explore diferentes filtros para gerar insights sobre a população de pacientes.")
