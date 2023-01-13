# Instalando bibliotecas que serão utilizadas neste projeto:
#!pip install matplotlib
#!pip install seaborn
#!pip install scikit.learn

# Importando a base de dados e analisando os dados:
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

tabela = pd.read_csv("advertising.csv")
print(tabela.head(10)) # verificando as 10 primeiras linhas 
print(tabela.info()) # verificando o tipo das variáveis entre outras informações

#Análise Exploratória
#print(f"Matriz Correlação \n {tabela.corr()}")
sns.heatmap(tabela.corr(), cmap="Wistia", annot= True)
sns.pairplot(tabela)
plt.show

# Usando agora a inteligencia artificial com o metodos de machine learning
# Vamos escolher o melhor modelo para prever o comportamento de novos dados:

y = tabela["Vendas"]
x = tabela[["TV", "Radio" ,"Jornal"]]
# O x poderia ser escrito apenas excluindo a coluna de Vendas: x = tabela.drop("Vendas", axis=1)

from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y, test_size=0.3)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# cria as inteligencias aritificiais
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

# Treinamento da inteligencia artificial
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

# Testando a IA 
testando_regressaolinear = modelo_regressaolinear.predict(x_teste)
testando_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

#Visualizando a previsão:
resultado = pd.DataFrame()
resultado["y_teste"] = y_teste
resultado["Previsoes da Regressao Linear"] = testando_regressaolinear
resultado["Previsoes da Árvore de decisão"] = testando_arvoredecisao
#print(resultado)

plt.figure(figsize=(15,6))
sns.lineplot(data=resultado)
plt.show()


#Avaliando as IA
from sklearn.metrics import r2_score
r2_regressaolinear = r2_score(y_teste,testando_regressaolinear)
r2_arvoredecisao = r2_score(y_teste,testando_arvoredecisao)
print(f"Comparando os modelos de Machine Learning, a metrica R2 Score da Regressão Linear foi de {r2_regressaolinear:.2%} e Arvore de decisão foi de {r2_arvoredecisao:.2%}")

# Prevendo um resultado de vendas novo em uma tabela só com dados de entrada:
nova_tabelax = pd.read_csv("prever_novos.csv")
print(nova_tabelax)
previsao_vendas = modelo_arvoredecisao.predict(nova_tabelax)
print(f'vendas: {previsao_vendas}')

# Analisando a importancia de cada variavel para com as vendas
sns.barplot(x=x_treino.columns, y=modelo_arvoredecisao.feature_importances_)
plt.show()

