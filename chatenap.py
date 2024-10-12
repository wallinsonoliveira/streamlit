import streamlit as st
import joblib
import gdown

# Link de download do Google Drive
file_id = '13x0B8bXcr9deS1Utda_t43__vbM1sy7S'  # Substitua pelo ID real do seu arquivo
download_url = f'https://drive.google.com/uc?id={file_id}'

# Baixar o arquivo
gdown.download(download_url, 'modelo_if_df.pkl', quiet=False)

# Carregar o modelo salvo
model = joblib.load('modelo_if_df.pkl')

# Configurações da página
st.set_page_config(page_title="Alunos Enap - Chat")
st.title("Alunos Enap - Chat")

# Campo de entrada para o texto
input_text = st.text_area("Digite uma frase:", "")

# Botão para fazer a predição
if st.button("Prever"):
    if input_text:
        # Aqui você deve ter a lógica de predição apropriada
        # Supondo que a predição use uma função 'predict' que você já tenha definido
        prediction = model.predict([input_text])  # Adapte conforme a entrada do modelo

        # Exibir o resultado
        st.success(f"Predição: {prediction[0]}")
    else:
        st.warning("Por favor, digite uma frase antes de prever.")
