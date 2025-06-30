import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pre_processamento import carregar_dados_processados

# Navegação lateral
opcoes = ['Boas-vindas', 'Dashboard']
pagina = st.sidebar.selectbox('Navegue pelo menu:', opcoes)

# Página de boas-vindas
if pagina == 'Boas-vindas':
    st.title("**🎈 Meu novo app**")
    st.write("Estou na página das Boas-vindas")

# Página do dashboard com dados processados
if pagina == 'Dashboard':
    st.title("📊 Dashboard - Dados Tratados")

    # with st.spinner("Processando os dados..."):
    #     dados = carregar_dados_processados()

    # st.success("Dados carregados com sucesso!")
    # st.subheader("Tabela Interativa:")
    # st.data_editor(dados)

    url = 'https://raw.githubusercontent.com/NandoBispo/global-cancer-patients/main/dados/global_cancer_patients_2015_2024.csv'
    df = pd.read_csv(url)
    #st.data_editor(df.sample(5))

    col1, col2, col3 = st.columns(3)

    st.markdown('---')

    # regiao = col1.selectbox("Região", dados['Country_Region'].unique())
    regiao = col1.selectbox("Região", df['Country_Region'].unique())
    sexo = col2.selectbox("Sexo", ['Masculino', 'Feminino'])
    ano = col3.selectbox("Ano", [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])

    sexo = 'Male' if sexo == 'Masculino' else 'Female'

    filtro_regiao = df['Country_Region'] == regiao
    filtro_sexo = df['Gender'] == sexo
    filtro_ano = df['Year'] == ano
    # filtro_regiao = dados['Country_Region'] == regiao
    # filtro_sexo = dados['Gender'] == sexo
    # filtro_ano = dados['Year'] == ano

    filtro_dados = df.loc[filtro_regiao & filtro_sexo & filtro_ano]
    # filtro_dados = dados.loc[filtro_regiao & filtro_sexo & filtro_ano]

    #st.table(filtro_dados.sample(5))

    col1, col2 = st.columns([1, 3])

    col1.metric('Idade Média', round(filtro_dados['Age'].mean(), 1))
    col1.metric('Tempo Médio de Vida', round(filtro_dados['Survival_Years'].mean(), 1))
    col1.metric('Custo Médio do Tratamento', round(filtro_dados['Treatment_Cost_USD'].mean(), 1))
    col1.metric('Estágio III do Câncer', '{:.2%}'.format(filtro_dados['Cancer_Stage'].value_counts(normalize = True)['Stage III']))
    col1.metric('Estágio IV do Câncer', '{:.2%}'.format(filtro_dados['Cancer_Stage'].value_counts(normalize = True)['Stage IV']))


    fig = sns.scatterplot(data = filtro_dados, x = 'Survival_Years', y = 'Treatment_Cost_USD', hue = 'Cancer_Stage')
    plt.xlabel('Anos de Sobrevivência')
    plt.ylabel('Custo do Tratamenro (USD)')

    col2.pyplot(fig.get_figure())

    st.markdown('---')