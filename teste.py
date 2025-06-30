import pickle

try:
    with open('modelo_cancer.pkl', 'rb') as file:
        modelo = pickle.load(file)
    print("Modelo carregado com sucesso!")
except Exception as e:
    print("Erro ao carregar modelo:", e)
