import streamlit as st
import joblib
import numpy as np

# Carregar o modelo salvo
model = joblib.load('iris_model.pkl')

# Função para prever a classe da flor com base nas características fornecidas
def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    return prediction[0]

# Interface do usuário no Streamlit
st.title("Classificação de Iris com Modelo Treinado")

# Inputs do usuário para as características da flor
sepal_length = st.number_input("Comprimento da Sépala (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Largura da Sépala (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Comprimento da Pétala (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Largura da Pétala (cm)", min_value=0.0, max_value=10.0, step=0.1)

# Botão para prever a classe
if st.button("Classificar"):
    # Fazer a previsão com o modelo
    prediction = predict_flower(sepal_length, sepal_width, petal_length, petal_width)
    
    # Exibir o resultado
    st.write(f"A classe da flor prevista é: {prediction}")
