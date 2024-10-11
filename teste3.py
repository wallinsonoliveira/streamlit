import streamlit as st
import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Título da aplicação
st.title("Classificação do Iris Dataset com Regressão Logística")

# Carregar o conjunto de dados Iris
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Criar um DataFrame
df = pd.DataFrame(data=X, columns=feature_names)
df['target'] = y

# Exibir os dados
st.subheader("Dados do Iris Dataset")
st.dataframe(df)

# Plotar o gráfico de dispersão
st.subheader("Gráfico de Dispersão das Classes")
sns.scatterplot(x=df[feature_names[0]], y=df[feature_names[1]], hue=df['target'], palette='Set1')
plt.title("Gráfico de Dispersão - Iris Dataset")
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
st.pyplot(plt)

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o modelo de Regressão Logística
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)


# Relatório de classificação
report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
st.write("Relatório de Classificação:")
st.text(classification_report(y_test, y_pred, target_names=target_names))

# Exibir a precisão média
accuracy = np.mean(y_pred == y_test)
st.write(f"Acurácia do Modelo: {accuracy:.2f}")
