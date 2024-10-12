import streamlit as st
import gdown
import joblib

# Título da página
st.title("Alunos Enap - Chat")

# Link de download do Google Drive
file_id = '13x0B8bXcr9deS1Utda_t43__vbM1sy7S'  # Substitua pelo ID real do seu arquivo
download_url = f'https://drive.google.com/uc?id={file_id}'

# Baixar o arquivo com tratamento de erros
try:
    gdown.download(download_url, 'modelo_if_df.pkl', quiet=False)
except Exception as e:
    st.error(f"Ocorreu um erro ao baixar o modelo: {e}")
    st.stop()  # Para a execução do script se o download falhar

# Carregar o modelo e o vetor salvo
try:
    model = joblib.load('modelo_if_df.pkl')
    vectorizer = joblib.load('vectorizer.pkl')  # Carregue o vetor correspondente
except Exception as e:
    st.error(f"Ocorreu um erro ao carregar o modelo ou vetor: {e}")
    st.stop()  # Para a execução do script se o carregamento falhar

# Campo para entrada de texto
input_text = st.text_input("Digite uma frase:")

if st.button("Prever"):
    # Use o vetor para transformar o texto antes da predição
    try:
        input_vector = vectorizer.transform([input_text])  # Transforme o texto
        prediction = model.predict(input_vector)  # Faça a predição
        st.write(f"O nome previsto é: {prediction[0]}")
    except Exception as e:
        st.error(f"Ocorreu um erro ao fazer a predição: {e}")

