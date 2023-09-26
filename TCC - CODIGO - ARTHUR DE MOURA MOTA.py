# Importando a biblioteca pandas para manipulação de dados
import pandas as pd

# Importando as bibliotecas matplotlib e seaborn para visualização de dados
import matplotlib.pyplot as plt
import seaborn as sns

# Configurando a paleta de cores do Seaborn para uma visualização agradável
sns.set_palette("viridis", 30)

# Definindo a paleta de cores do Seaborn como um mapa de cores (cmap) para uso posterior
sns.color_palette("viridis", as_cmap=True)

# Importando bibliotecas do scikit-learn para preparação de dados e modelagem
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor  # Importando o modelo GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Importando a biblioteca statsmodels para análise estatística
import statsmodels.api as sm

# Importando a função ols (Ordinary Least Squares) do statsmodels para ajuste de modelos lineares
from statsmodels.formula.api import ols

# Exibe todas as colunas do DataFrame sem truncamento
pd.set_option('display.max_columns', None)

# Configuração para exibir os números com precisão fixa de 6 casas decimais
pd.set_option('display.float_format', '{:.4f}'.format)

pd.set_option('display.max_rows', None)

def analisar_ANOVA(df,colunas_categoricas):
    # Criar uma fórmula para a ANOVA
    formula_anova = 'VR_DESPESA_MAX_CAMPANHA ~ ' + ' + '.join(['C(' + coluna + ')' for coluna in colunas_categoricas])
    modelo_anova = ols(formula_anova, data=df).fit()

    tabela_anova = sm.stats.anova_lm(modelo_anova, typ=2)

    # Exibir os resultados da ANOVA
    print(f"Resultados da Análise de Variância (ANOVA) para as colunas: {', '.join(colunas_categoricas)}")
    print(tabela_anova)

# Função para substituir os valores da lista 'populacao_estimada_correcao' por uma string vazia
def substituir_valor (valor):
    for item in populacao_estimada_correcao:
        valor = valor.replace(item, '')
        valor = valor.replace(" ", "")

    return valor



def plot_boxplot(df, nome_df, coluna_nome):
    # Normaliza os dados da coluna
    coluna_normalizada = (df[coluna_nome] - df[coluna_nome].mean()) / df[coluna_nome].std()

    # Cria um boxplot da coluna normalizada usando Seaborn
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=coluna_normalizada, color='red', flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markeredgecolor': 'red', 'markersize': 5})

    # Configurações adicionais do gráfico
    plt.title(f'Boxplot da Coluna {coluna_nome} (Normalizada) - {nome_df}')
    plt.ylabel(coluna_nome + ' (Normalizada)')
    plt.ticklabel_format(axis='y', style='plain', useOffset=False)

    # Mostra o gráfico
    plt.show()


def plot_boxplots(df, colunas_numericas, df_nome):
    # Calcula o número de subplots com base no número de colunas numéricas
    num_plots = len(colunas_numericas)

    # Define o número de colunas para a disposição dos subplots (pode ajustar conforme necessário)
    num_cols = 1
    num_rows = (num_plots + 1)   # Garante pelo menos uma linha

    # Cria subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20,  num_rows))
    fig.suptitle(f'Boxplots das Colunas Numéricas em {df_nome}', fontsize=16)

    # Certifique-se de que axes seja uma lista de eixos (axis) mesmo para um único subplot
    if num_rows == 1:
        axes = [axes]
    # Configuração personalizada para os outliers
    flierprops = dict(marker='o', markerfacecolor='red', markeredgecolor= 'red' , markersize=2, linestyle='none')

    for i, col in enumerate(colunas_numericas):
        ax = axes[i]
        sns.boxplot(data=df, x=col, orient="horizontal", ax=ax, flierprops= flierprops)
        ax.set_title(col)
        ax.set_xlabel(None)
        ax.set_xticks([])

    # Remove subplots vazios, se houverem
    for i in range(len(colunas_numericas), num_rows * num_cols):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def plotar_matriz_correlacao(df,colunas_numericas, nome_df):


    # Selecionar apenas as colunas numéricas do DataFrame
    numeric_columns = df.select_dtypes(include=['int64', 'float64'])


    # Calcular a matriz de correlação de Pearson
    correlation_matrix = df[colunas_numericas].corr()
    print(nome_df + " :")
    print(correlation_matrix['VR_DESPESA_MAX_CAMPANHA'])

    # Plotar a matriz de correlação
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",linewidths=0.5)
    plt.title(nome_df + ' Matriz de Correlação (Pearson) entre Variáveis numéricas e VR_DESPESA_MAX_CAMPANHA')
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    #tick_params(axis='x', rotation=90)
    plt.show()

def remover_outliers_VR_DESPESA_MAX_CAMPANHA(df,nome_df):
    # Calcula a média e o desvio padrão da coluna 'VR_DESPESA_MAX_CAMPANHA'
    media_despesa = df['VR_DESPESA_MAX_CAMPANHA'].mean()
    desvio_padrao_despesa = df['VR_DESPESA_MAX_CAMPANHA'].std()

    # Define o limiar para identificar outliers usando a regra 3-sigma
    limiar_superior = media_despesa + 3 * desvio_padrao_despesa
    limiar_inferior = media_despesa - 3 * desvio_padrao_despesa

    # Filtra os outliers em um novo DataFrame
    df_outliers = df[(df['VR_DESPESA_MAX_CAMPANHA'] < limiar_inferior) |
                        (df['VR_DESPESA_MAX_CAMPANHA'] > limiar_superior)]

    # Filtra os dados originais removendo os outliers
    df = df[(df['VR_DESPESA_MAX_CAMPANHA'] >= limiar_inferior) &
             (df['VR_DESPESA_MAX_CAMPANHA'] <= limiar_superior)]


    # Exiba informações sobre os outliers
    print(f"Número de outliers encontrados em {nome_df}: {len(df_outliers['VR_DESPESA_MAX_CAMPANHA'])}")
    print(f"Percentagem de outliers em {nome_df}:"
          f" {len(df_outliers['VR_DESPESA_MAX_CAMPANHA'])  / len(df['VR_DESPESA_MAX_CAMPANHA']) * 100:.2f}%")

    # Exibe estatísticas descritivas após remover os outliers
    print("Estatísticas descritivas após remover os outliers:")
    print(df['VR_DESPESA_MAX_CAMPANHA'].describe())

    # O DataFrame df agora contém os dados sem outliers e df_outliers apenas os outliers
    return [df,df_outliers]

def calcular_melhores_parametros(X,y):
    # Defina os hiperparâmetros que você deseja otimizar
    param_grid = {
        'n_estimators': [200,300],
        'learning_rate': [0.1],
        'max_depth': [4,6]
    }

    # Crie uma instância do regressor GradientBoosting
    regressor = GradientBoostingRegressor()

    # Crie um objeto GridSearchCV
    grid_search = GridSearchCV(estimator=regressor, param_grid=param_grid, cv=2, n_jobs=-1,verbose=4)

    # Realize a pesquisa em grade
    grid_search.fit(X, y)

    # Exiba os melhores hiperparâmetros encontrados
    print("Melhores hiperparâmetros encontrados:")
    print(grid_search.best_params_)

    # Exiba a melhor pontuação do modelo
    print("Melhor pontuação do modelo:")
    print(grid_search.best_score_)

    # Ajuste o modelo final com os melhores hiperparâmetros
    melhor_modelo = grid_search.best_estimator_
    return [melhor_modelo]

def analise_de_modelo_VR_DESPESA_MAX_CAMPANHA(df,nomedf,analise_resisduos):

    # Seleciona as colunas de recursos numéricos
    X_colunas_numericas = df[[
        'VR_DESPESA_MAX_CAMPANHA',
        'POPULAÇÃO ESTIMADA',
        'Valor adicionado bruto da Agropecuária, a preços correntes (R$ 1.000)',
        'Valor adicionado bruto da Indústria, a preços correntes (R$ 1.000)',
        'Valor adicionado bruto dos Serviços, a preços correntes - exceto Administração, defesa, educação e saúde públicas e seguridade social (R$ 1.000)',
        'Valor adicionado bruto da Administração, defesa, educação e saúde públicas e seguridade social, a preços correntes (R$ 1.000)',
        'Valor adicionado bruto total, a preços correntes (R$ 1.000)',
        'Impostos, líquidos de subsídios, sobre produtos, a preços correntes (R$ 1.000)',
        'Produto Interno Bruto per capita, a preços correntes (R$ 1,00)']]

    # Cria variáveis dummy para as colunas categóricas
    X_colunas_categoricas = pd.get_dummies(df[[
        'DS_CARGO',
        'TP_AGREMIACAO',
        'SG_PARTIDO',
        'ST_REELEICAO',
        'capital',
        'UF',
        'Nome_da_Grande_Região'


    ]], drop_first=True)

    # Combina as variáveis numéricas e categóricas codificadas
    X = pd.concat([X_colunas_numericas, X_colunas_categoricas], axis=1)


    # Define a variável alvo
    y = df['VR_DESPESA_MAX_CAMPANHA']

    # Divide o conjunto de dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treina o modelo e obtém os melhores parâmetros
    modelo = calcular_melhores_parametros(X_train,y_train)

    # Ajusta o modelo aos dados de treinamento
    modelo[0].fit(X_train, y_train)

    # Faz previsões no conjunto de teste
    y_pred = modelo[0].predict(X_test)

    # Avalia o desempenho do modelo

    # Calcular o Erro Quadrático Médio (MSE)
    erro_quadratico_medio = mean_squared_error(y_test, y_pred)
    print(f"Erro Quadrático Médio (MSE): {erro_quadratico_medio}")

    # Calcular o Erro Absoluto Médio (MAE)
    erro_absoluto_medio = mean_absolute_error(y_test, y_pred)
    print(f"Erro Absoluto Médio (MAE): {erro_absoluto_medio}")

    # Calcular o Coeficiente de Determinação (R²)
    coeficiente_determinacao = r2_score(y_test, y_pred)
    print(f"Coeficiente de Determinação (R²): {coeficiente_determinacao}")

    # Calcular a Raiz do Erro Quadrático Médio (RMSE)
    raiz_erro_quadratico_medio = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Raiz do Erro Quadrático Médio (RMSE): {raiz_erro_quadratico_medio}")
    print("\n")

    if analise_resisduos == "S":
        # Calcula os resíduos do modelo
        residuos = y_test - y_pred

        # Cria um DataFrame para os resíduos
        residuos_df = pd.DataFrame({'Previsões': y_pred, 'Resíduos': residuos})

        # Gráfico de dispersão dos resíduos em relação às previsões usando Seaborn
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=residuos_df, x='Previsões', y='Resíduos', color='green', alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('Previsões')
        plt.ylabel('Resíduos')
        plt.title(f'Gráfico de Dispersão dos Resíduos em Função das Previsões - {nomedf}')
        plt.ticklabel_format(axis='y', style='plain', useOffset=False)
        plt.ticklabel_format(axis='x', style='plain', useOffset=False)

        plt.show()


# Função para criar um gráfico de donut a partir dos dados fornecidos
def plot_donut(df, titulo, ax):
    # Extrai os rótulos e tamanhos dos dados
    labels = df['ST_REELEICAO']
    sizes = df['VR_DESPESA_MAX_CAMPANHA']

    # Cria o gráfico de pizza (donut)
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    ax.set_title(titulo)

    # Adiciona um círculo no centro para criar o efeito de donut
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax.add_artist(centre_circle)


print('*' * 100)
print('INICIO Coleta de Dados - Processamento/Tratamento de Dados')
print('*' * 100)
print('*' * 100)
print('INICIO df_TSE_IBGE')
print('*' * 100)

# Carrega o DataFrame com dados do TSE e IBGE
df_TSE_IBGE = pd.read_csv('DATASETS/municipios_brasileiros_tse.csv', encoding='UTF-8', delimiter=',')

# Imprime as primeiras linhas do DataFrame
print(df_TSE_IBGE.head())
# Imprime informações do DataFrame
print(df_TSE_IBGE.info())
# Imprime estatísticas descritivas do DataFrame
print(df_TSE_IBGE.describe())
# Converte colunas específicas para o tipo de dado 'str'
df_TSE_IBGE['codigo_ibge'] = df_TSE_IBGE['codigo_ibge'].astype(str)
df_TSE_IBGE['codigo_tse'] = df_TSE_IBGE['codigo_tse'].astype(str)

print('*' * 100)
print('FIM df_TSE_IBGE')
print('*' * 100)



print('*' * 100)
print('INICIO df_eleitos')
print('*' * 100)

df_eleitos = pd.read_csv('DATASETS/consulta_cand_2020_BRASIL.csv', encoding='latin1', delimiter=';')
# Exibe as colunas do dataframe e informações básicas
print("Colunas do df_eleitos:")
print(df_eleitos.columns)
print("\nInformações do df_eleitos:")
print(df_eleitos.info())

# Lista das colunas desejadas
eleitos_colunas_desejadas = [
    'CD_TIPO_ELEICAO',
    'NM_TIPO_ELEICAO',
    'SG_UF',
    'SG_UE',
    'NM_UE',
    'CD_CARGO',
    'DS_CARGO',
    'CD_SITUACAO_CANDIDATURA',
    'DS_SITUACAO_CANDIDATURA',
    'TP_AGREMIACAO',
    'SG_PARTIDO',
    'NM_PARTIDO',
    'CD_SIT_TOT_TURNO',
    'DS_SIT_TOT_TURNO',
    'ST_REELEICAO',
    'VR_DESPESA_MAX_CAMPANHA'
]

# Filtra o dataframe para manter apenas as colunas desejadas
df_eleitos = df_eleitos[eleitos_colunas_desejadas]

# Converte a coluna 'SG_UE' para o tipo string, se necessário
df_eleitos['SG_UE'] = df_eleitos['SG_UE'].astype(str)

# Exibe o número de valores únicos em cada coluna
print("\nNúmero de valores únicos em cada coluna:")
print(df_eleitos.nunique())


# Lista das colunas a serem analisadas
colunas_para_analisar = [
    'CD_TIPO_ELEICAO',
    'NM_TIPO_ELEICAO',
    'CD_CARGO',
    'DS_CARGO',
    'CD_SITUACAO_CANDIDATURA',
    'DS_SITUACAO_CANDIDATURA',
    'TP_AGREMIACAO',
    'SG_PARTIDO',
    'NM_PARTIDO',
    'CD_SIT_TOT_TURNO',
    'DS_SIT_TOT_TURNO',
    'ST_REELEICAO'
]

# Itera sobre as colunas para exibir os valores únicos em cada uma
for coluna in colunas_para_analisar:
    valores_unicos = df_eleitos[coluna].unique()
    print(f"Valores únicos em {coluna}:\n{valores_unicos}\n")

# Lista dos valores de 'DS_SIT_TOT_TURNO' desejados
eleitos_selecionados_eleito = ['ELEITO', 'ELEITO POR QP', 'ELEITO POR MÉDIA']

# Filtra o dataframe para manter apenas as linhas com as situações desejadas
df_eleitos = df_eleitos[df_eleitos['DS_SIT_TOT_TURNO'].isin(eleitos_selecionados_eleito)]

# Exibe informações do dataframe após a primeira filtragem
print("\nInformações do dataframe após a primeira filtragem DS_SIT_TOT_TURNO:")
print(df_eleitos.info())


# Lista dos valores de 'DS_SITUACAO_CANDIDATURA' desejados
eleitos_selecionados_apto = ['APTO']

# Filtra o dataframe para manter apenas as linhas com as situações de candidatura desejadas
df_eleitos = df_eleitos[df_eleitos['DS_SITUACAO_CANDIDATURA'].isin(eleitos_selecionados_apto)]

# Exibe informações do dataframe após a segunda filtragem
print("\nInformações do dataframe após a segunda filtragem DS_SITUACAO_CANDIDATURA:")
print(df_eleitos.info())

# Lista das colunas a serem removidas
colunas_para_remover = [
    "NM_UE",
    "CD_CARGO",
    "CD_TIPO_ELEICAO",
    "NM_TIPO_ELEICAO",
    "CD_SITUACAO_CANDIDATURA",
    "DS_SITUACAO_CANDIDATURA",
    "CD_SIT_TOT_TURNO",
    "DS_SIT_TOT_TURNO"
]


# Remove as colunas especificadas do dataframe
df_eleitos = df_eleitos.drop(columns=colunas_para_remover)

# Exibe informações do dataframe após a remoção de colunas
print("\nInformações do dataframe após a remoção de colunas:")
print(df_eleitos.info())


print('*' * 100)
print('FIM df_eleitos')
print('*' * 100)


print('*' * 100)
print('INICIO df_populacao')
print('*' * 100)
# Carregar o DataFrame a partir do arquivo CSV
# Usar skiprows=1 para ignorar o cabeçalho
df_populacao = pd.read_excel('DATASETS/estimativa_dou_2020.xls',
                             skiprows=1, header=0, sheet_name='Municípios')

# Exibir as 10 primeiras linhas das colunas 'COD. UF' e 'COD. MUNIC'
print(df_populacao['COD. UF'].head(10))
print(df_populacao['COD. MUNIC'].head(10))

# Identificar as colunas categóricas e numéricas
categorical_columns = df_populacao.select_dtypes(include=['object']).columns
numeric_columns = df_populacao.select_dtypes(include=['int', 'float']).columns


# Análise das variáveis categóricas
print('Análise das variáveis categóricas')

for col in categorical_columns:
    print(f"Análise de {col}:")
    print(f"Total de valores únicos: {df_populacao[col].nunique()}")
    print(f"Valores únicos: {df_populacao[col].unique()}")
    print("===")

print('# Análise das variáveis numéricas')
# Análise das variáveis numéricas
for col in numeric_columns:
    print(f"Análise de {col}:")
    print(f"Média: {df_populacao[col].mean()}")
    print(f"Desvio padrão: {df_populacao[col].std()}")
    print(f"Valor mínimo: {df_populacao[col].min()}")
    print(f"Valor máximo: {df_populacao[col].max()}")
    print("===")

# Exibir informações gerais sobre o DataFrame
print(df_populacao.info())

# Filtrar as linhas onde a coluna 'UF' está na lista 'uf_correcao'
uf_correcao = ['RO', 'AC', 'AM', 'RR', 'PA', 'AP',
               'TO', 'MA', 'PI', 'CE', 'RN', 'PB',
               'PE', 'AL', 'SE', 'BA', 'MG', 'ES',
               'RJ', 'SP', 'PR', 'SC', 'RS', 'MS',
               'MT', 'GO', 'DF']

# Filtre as linhas onde a coluna 'UF' está na lista 'uf_correcao'.
df_populacao = df_populacao[df_populacao['UF'].isin(uf_correcao)]

# Análise da coluna 'UF' após a filtragem
print(f"Análise de 'UF':")
print(f"Total de valores únicos: {df_populacao['UF'].nunique()}")
print(f"Valores únicos: {df_populacao['UF'].unique()}")
print("===")

# Exibir informações atualizadas sobre o DataFrame após a filtragem
print(df_populacao.info())
# Exibir estatísticas descritivas do DataFrame
print(df_populacao.describe())


# Tratar a coluna 'POPULAÇÃO ESTIMADA'
# Use a função `pd.to_numeric` com `errors='coerce'` para converter a coluna em números
# e lidar com valores não numéricos.
valores_nao_numericos = df_populacao[df_populacao['POPULAÇÃO ESTIMADA'].apply(pd.to_numeric, errors='coerce').isna()]

# Exibir os registros não numéricos
print(valores_nao_numericos)

# Lista de índices de valores não numéricos
lista_valores_nao_numericos = valores_nao_numericos.index.tolist()
print(lista_valores_nao_numericos)

# Valores a serem corrigidos na coluna 'POPULAÇÃO ESTIMADA'
populacao_estimada_correcao = ['(1)', '(2)', '(3)', '(4)',
                               '(5)', '(6)', '(7)', '(8)',
                               '(9)', '(10)', '(11)', '(12)']




# Aplicar a substituição apenas nos índices especificados em 'lista_valores_nao_numericos'
df_populacao.loc[lista_valores_nao_numericos, 'POPULAÇÃO ESTIMADA'] = df_populacao.loc[
    lista_valores_nao_numericos, 'POPULAÇÃO ESTIMADA'].apply(substituir_valor)
print("df_populacao.loc[lista_valores_nao_numericos, 'POPULAÇÃO ESTIMADA']")
print(df_populacao.loc[lista_valores_nao_numericos])

# Usar a função pd.to_numeric para converter a coluna em números, definindo errors='coerce'
# para tratar não numéricos como NaN.
df_populacao['POPULAÇÃO ESTIMADA'] = pd.to_numeric(df_populacao['POPULAÇÃO ESTIMADA'], errors='coerce')

# Converter as colunas para o tipo de dados 'strings'
df_populacao['UF'] = df_populacao['UF'].astype(str)
df_populacao['COD. UF'] = df_populacao['COD. UF'].astype(str).str.split('.').str[0]

# Converter a coluna 'COD. MUNIC' em strings, dividir por '.', pegar a parte inteira
# e aplicar o preenchimento com zeros à esquerda
df_populacao['COD. MUNIC'] = df_populacao['COD. MUNIC'].astype(str)
df_populacao['COD. MUNIC'] = df_populacao['COD. MUNIC'].apply(lambda x: str(x).replace('.', '').zfill(6))

df_populacao['NOME DO MUNICÍPIO'] = df_populacao['NOME DO MUNICÍPIO'].astype(str)

# Verificar os tipos de dados atualizados
print(df_populacao.dtypes)

# Exibir estatísticas descritivas atualizadas do DataFrame
print(df_populacao.describe())
# Exibir informações atualizadas sobre o DataFrame
print(df_populacao.info())
# Exibir as colunas do DataFrame
print(df_populacao.columns)

print('*' * 100)
print('FIM df_populacao')
print('*' * 100)

print('*' * 100)
print('INICIO df_renda')
print('*' * 100)

# Carrega o conjunto de dados do PIB dos Municípios
df_renda = pd.read_excel('DATASETS/PIB dos Municípios - base de dados 2010-2020.xls',
                         skiprows=0, header=0, sheet_name='PIB_dos_Municípios')

# Exibe as colunas do DataFrame
print("Colunas do df_renda:")
print(df_renda.columns)
# Exibe informações sobre o DataFrame
print("Informações do df_renda:")
print(df_renda.info())


# Filtra o DataFrame para manter apenas os dados do ano de 2020
df_renda = df_renda[df_renda['Ano'] == 2020]

# Exibe informações atualizadas do DataFrame
print("Informações do DataFrame após filtragem:")
print(df_renda.info())



# Seleciona as colunas de interesse
renda_colunas_selecionadas = [
    'Nome da Grande Região',
    'Código do Município',
    'Valor adicionado bruto da Agropecuária, \na preços correntes\n(R$ 1.000)',
    'Valor adicionado bruto da Indústria,\na preços correntes\n(R$ 1.000)',
    'Valor adicionado bruto dos Serviços,\na preços correntes \n- exceto Administração, defesa, educação e saúde públicas e seguridade social\n(R$ 1.000)',
    'Valor adicionado bruto da Administração, defesa, educação e saúde públicas e seguridade social, \na preços correntes\n(R$ 1.000)',
    'Valor adicionado bruto total, \na preços correntes\n(R$ 1.000)',
    'Impostos, líquidos de subsídios, sobre produtos, \na preços correntes\n(R$ 1.000)',
    'Produto Interno Bruto per capita, \na preços correntes\n(R$ 1,00)'
]
# Mantém apenas as colunas de interesse no DataFrame
df_renda = df_renda[renda_colunas_selecionadas]
# Renomeia as colunas para nomes mais legíveis
df_renda.rename(columns={
    'Valor adicionado bruto da Agropecuária, \na preços correntes\n(R$ 1.000)':
        'Valor adicionado bruto da Agropecuária, a preços correntes (R$ 1.000)',

    'Valor adicionado bruto da Indústria,\na preços correntes\n(R$ 1.000)':
        'Valor adicionado bruto da Indústria, a preços correntes (R$ 1.000)',

    'Valor adicionado bruto dos Serviços,\na preços correntes \n- exceto Administração, '
    'defesa, educação e saúde públicas e seguridade social\n(R$ 1.000)':
        'Valor adicionado bruto dos Serviços, a preços correntes - exceto Administração, defesa,'
        ' educação e saúde públicas e seguridade social (R$ 1.000)',

    'Valor adicionado bruto da Administração, defesa, educação e saúde públicas e seguridade social,'
    ' \na preços correntes\n(R$ 1.000)':
        'Valor adicionado bruto da Administração, defesa, educação e saúde públicas e seguridade social,'
        ' a preços correntes (R$ 1.000)',

    'Valor adicionado bruto total, \na preços correntes\n(R$ 1.000)':
        'Valor adicionado bruto total, a preços correntes (R$ 1.000)',

    'Impostos, líquidos de subsídios, sobre produtos, \na preços correntes\n(R$ 1.000)':
        'Impostos, líquidos de subsídios, sobre produtos, a preços correntes (R$ 1.000)',

    'Produto Interno Bruto per capita, \na preços correntes\n(R$ 1,00)':
        'Produto Interno Bruto per capita, a preços correntes (R$ 1,00)'
}, inplace=True)

# Renomeia a coluna 'Nome da Grande Região'
df_renda = df_renda.rename(columns={'Nome da Grande Região': 'Nome_da_Grande_Região'})


# Converte a coluna 'Código do Município' para tipo string
df_renda['Código do Município'] = df_renda['Código do Município'].astype(str)

# Exibe informações atualizadas do DataFrame
print("Informações do df_renda após seleção e renomeação de colunas:")
print(df_renda.info())

# Exibe estatísticas descritivas do DataFrame
print("Estatísticas Descritivas do df_renda:")
print(df_renda.describe())

print('*' * 100)
print('FIM df_renda')
print('*' * 100)

print('*' * 100)
print('Inicio df_dados_demograficos = df_renda + df_populacao')
print('*' * 100)

# Cria uma coluna temporária em ambos os DataFrames para facilitar a junção
df_populacao['COD. UF + COD. MUNIC'] = (df_populacao['COD. UF'] + df_populacao['COD. MUNIC'].astype(str)).str[:-1]
df_populacao['COD. UF + COD. MUNIC'] = df_populacao['COD. UF + COD. MUNIC'].astype(str)
print("Exemplo de dados nas colunas 'COD_UF', 'COD_MUNIC' e 'COD. UF + COD. MUNI':")
print(df_populacao['COD. UF'].head(5))
print(df_populacao['COD. MUNIC'].head(5))
print(df_populacao['COD. UF + COD. MUNIC'].head(5))


# Seleciona as colunas desejadas em cada DataFrame
colunas_df_populacao = ['UF', 'NOME DO MUNICÍPIO', 'POPULAÇÃO ESTIMADA']
colunas_df_renda = df_renda.columns.tolist()



# Realiza a junção entre os DataFrames usando a coluna temporária como chave
df_dados_demograficos = pd.merge(df_renda, df_populacao, left_on='Código do Município', right_on='COD. UF + COD. MUNIC', how='inner')[
    colunas_df_renda + colunas_df_populacao]


print("Informações sobre o df_dados_demograficos:")
print(df_dados_demograficos.info())

# Imprime as 10 primeiras linhas do DataFrame resultante
print("As 5 Primeiras Linhas do DataFrame Combinado:")
print(df_dados_demograficos.head(5))

print('*' * 100)
print('FIM df_dados_demograficos = df_renda + df_populacao')
print('*' * 100)


print('*' * 100)
print('Início df_dados_demograficos completo')
print('*' * 100)

# Define a ordem desejada das colunas em uma lista
nova_ordem_colunas = [  'NOME DO MUNICÍPIO',
	   'Código do Município',
	   'UF',
	   'Nome_da_Grande_Região',
	   'POPULAÇÃO ESTIMADA',
	   'Valor adicionado bruto da Agropecuária, a preços correntes (R$ 1.000)',
       'Valor adicionado bruto da Indústria, a preços correntes (R$ 1.000)',
       'Valor adicionado bruto dos Serviços, a preços correntes - exceto Administração, defesa, educação e saúde públicas e seguridade social (R$ 1.000)',
       'Valor adicionado bruto da Administração, defesa, educação e saúde públicas e seguridade social, a preços correntes (R$ 1.000)',
       'Valor adicionado bruto total, a preços correntes (R$ 1.000)',
       'Impostos, líquidos de subsídios, sobre produtos, a preços correntes (R$ 1.000)',
       'Produto Interno Bruto per capita, a preços correntes (R$ 1,00)'

]

# Reordena as colunas do DataFrame
df_dados_demograficos = df_dados_demograficos[nova_ordem_colunas]
print('Nomes das colunas após reordenamento:')
print(df_dados_demograficos.columns)

# Imprime informações sobre o DataFrame
print('Informações sobre o DataFrame:')
print(df_dados_demograficos.info())


print('*' * 100)
print('FIM df_dados_demograficos completo')
print('*' * 100)

print('*' * 100)
print('Início df_eleitos')
print('*' * 100)

# Selecionando as colunas desejadas para o DataFrame df_eleitos
colunas_df_eleitos = [
                      'DS_CARGO',
                      'TP_AGREMIACAO',
                      'SG_PARTIDO',
                      'NM_PARTIDO',
                      'ST_REELEICAO',
                      'VR_DESPESA_MAX_CAMPANHA']
# Selecionando as colunas desejadas para o DataFrame df_TSE_IBGE
colunas_df_TSE_IBGE = ['capital', 'codigo_ibge']

# Realizando a junção entre os DataFrames usando a coluna 'SG_UE' e 'codigo_tse' como chave
df_eleitos_IBGE = pd.merge(df_eleitos, df_TSE_IBGE, left_on='SG_UE', right_on='codigo_tse',
                           how='inner')[colunas_df_eleitos + colunas_df_TSE_IBGE]

# Campos preenchidos com #NULO significam que a informação está em branco no banco de dados.
# O correspondente para #NULO nos campos numéricos é -1;

# Separando os dados em duas categorias: >= 0 e < 0
valores_positivos = df_eleitos_IBGE[df_eleitos_IBGE['VR_DESPESA_MAX_CAMPANHA'] >= 0]
valores_negativos = df_eleitos_IBGE[df_eleitos_IBGE['VR_DESPESA_MAX_CAMPANHA'] < 0]

# Contando os valores em cada categoria
contagem_positivos = len(valores_positivos)
contagem_negativos = len(valores_negativos)

print("Contagem de Valores de VR_DESPESA_MAX_CAMPANHA")
print(f"Contagem_positivos: {contagem_positivos}")
print(f"Contagem_negativos: {contagem_negativos}")


# Criando um gráfico de barras para visualizar a contagem
categorias = ['Valores >= 0', 'Valores < 0']
contagem = [contagem_positivos, contagem_negativos]

# Cria um gráfico de barras usando Seaborn
sns.barplot(x=categorias, y=contagem)

# Define rótulos e título
plt.xlabel('Categorias')
plt.ylabel('Contagem')
plt.title('Contagem de Valores de VR_DESPESA_MAX_CAMPANHA')
plt.ticklabel_format(axis='y', style='plain', useOffset=False)

# Exibe o gráfico
plt.show()

# Filtrando apenas os dados relacionados aos vice-prefeitos antes da remoção de valores negativos
print("DataFrame df_eleitos_vices antes da remoção de valores negativos:")
df_eleitos_vices = df_eleitos_IBGE[df_eleitos_IBGE['DS_CARGO'] == 'VICE-PREFEITO']
print(df_eleitos_vices.info())

# Removendo valores negativos do DataFrame df_eleitos_com_IBGE
df_eleitos_IBGE = df_eleitos_IBGE[df_eleitos_IBGE['VR_DESPESA_MAX_CAMPANHA'] >= 0]

# Filtrando apenas os dados relacionados aos vice-prefeitos após a remoção de valores negativos
print("DataFrame df_eleitos_vices depois da remoção de valores negativos:")
df_eleitos_vices = df_eleitos_IBGE[df_eleitos_IBGE['DS_CARGO'] == 'VICE-PREFEITO']
print(df_eleitos_vices.info())

# Exibindo informações sobre o DataFrame df_eleitos_IBGE
print("Informações sobre o DataFrame df_eleitos_IBGE:")
print(df_eleitos_IBGE.info())


print('*' * 100)
print('FIM df_eleitos')
print('*' * 100)

print('*' * 100)
print('INICIO df_eleitos_dados_demograficos = df_eleitos_IBGE + df_dados_demograficos')
print('*' * 100)

# Obtém as listas de colunas dos DataFrames
colunas_df_eleitos_IBGE= df_eleitos_IBGE.columns.tolist()
colunas_df_dados_demograficos = df_dados_demograficos.columns.tolist()


# Realiza a junção entre os DataFrames
df_eleitos_dados_demograficos = pd.merge(df_eleitos_IBGE, df_dados_demograficos, left_on='codigo_ibge',
                                         right_on='Código do Município',
                                         how='inner')[colunas_df_eleitos_IBGE + colunas_df_dados_demograficos]

# Remove a coluna 'codigo_ibge'
df_eleitos_dados_demograficos = df_eleitos_dados_demograficos.drop(columns='codigo_ibge')

# Lista de colunas para definir como categóricas
colunas_categoricas = [
    'DS_CARGO',
    'TP_AGREMIACAO',
    'SG_PARTIDO',
    'NM_PARTIDO',
    'ST_REELEICAO',
    'capital',
    'UF',
    'Nome_da_Grande_Região'
   ]


# Define as colunas como categóricas
df_eleitos_dados_demograficos[colunas_categoricas] = df_eleitos_dados_demograficos[colunas_categoricas].astype('category')

# Lista de colunas para definir como numéricas
colunas_numericas = [
    'VR_DESPESA_MAX_CAMPANHA',
    'POPULAÇÃO ESTIMADA',
    'Valor adicionado bruto da Agropecuária, a preços correntes (R$ 1.000)',
    'Valor adicionado bruto da Indústria, a preços correntes (R$ 1.000)',
    'Valor adicionado bruto dos Serviços, a preços correntes - exceto Administração, defesa, educação e saúde públicas '
    'e seguridade social (R$ 1.000)',
    'Valor adicionado bruto da Administração, defesa, educação e saúde públicas e seguridade social, a preços correntes'
    ' (R$ 1.000)',
    'Valor adicionado bruto total, a preços correntes (R$ 1.000)',
    'Impostos, líquidos de subsídios, sobre produtos, a preços correntes (R$ 1.000)',
    'Produto Interno Bruto per capita, a preços correntes (R$ 1,00)'
]


# Define a ordem desejada das colunas
nova_ordem_colunas = [
    'NOME DO MUNICÍPIO',
    'Código do Município',
    'UF',
    'Nome_da_Grande_Região',
    'POPULAÇÃO ESTIMADA',
    'capital',
    'DS_CARGO',
    'VR_DESPESA_MAX_CAMPANHA',
    'SG_PARTIDO',
    'NM_PARTIDO',
    'TP_AGREMIACAO',
    'ST_REELEICAO',
    'Valor adicionado bruto da Agropecuária, a preços correntes (R$ 1.000)',
    'Valor adicionado bruto da Indústria, a preços correntes (R$ 1.000)',
    'Valor adicionado bruto dos Serviços, a preços correntes - exceto Administração, defesa, educação e saúde públicas e'
    ' seguridade social (R$ 1.000)',
    'Valor adicionado bruto da Administração, defesa, educação e saúde públicas e seguridade social, a preços correntes '
    '(R$ 1.000)',
    'Valor adicionado bruto total, a preços correntes (R$ 1.000)',
    'Impostos, líquidos de subsídios, sobre produtos, a preços correntes (R$ 1.000)',
    'Produto Interno Bruto per capita, a preços correntes (R$ 1,00)'
]

# Reordena as colunas do DataFrame
df_eleitos_dados_demograficos = df_eleitos_dados_demograficos[nova_ordem_colunas]

# Usa pd.to_numeric para definir as colunas como numéricas
df_eleitos_dados_demograficos[colunas_numericas] = df_eleitos_dados_demograficos[colunas_numericas].apply(pd.to_numeric,
                                                                                                          errors='coerce')


# Ordenar o DataFrame com base em várias colunas
df_eleitos_dados_demograficos = df_eleitos_dados_demograficos.sort_values(by=['UF', 'NOME DO MUNICÍPIO', 'SG_PARTIDO',
                                                                              'DS_CARGO'], ascending=[True, True, True, True])


# Imprime informações sobre o DataFrame após a definição dos tipos de colunas
print("Informações sobre o df_eleitos_dados_demograficos após a definição dos tipos das colunas:")
print(df_eleitos_dados_demograficos.info())


# Verifica os tipos de dados das colunas após a conversão
print("Tipos de dados das colunas após a conversão:")
print(df_eleitos_dados_demograficos.dtypes)


# Realiza uma análise exploratória básica
# Exibe algumas linhas aleatórias do DataFrame
print("Amostra aleatória do df_eleitos_dados_demograficos:")
print(df_eleitos_dados_demograficos.sample(5))


print('*' * 100)
print('FIM df_eleitos_dados_demograficos = df_eleitos_IBGE + df_dados_demograficos')
print('*' * 100)

print('*' * 100)
print('FIM Coleta de Dados - Processamento/Tratamento de Dados')
print('*' * 100)

print('*' * 100)
print('INICIO Análise e Exploração dos Dados - Criação de Modelos de Machine Learning')
print('*' * 100)


# Criar DataFrames df_eleitos_prefeitos  e df_eleitos_vereadores separados com base na coluna 'DS_CARGO'
df_eleitos_prefeitos = df_eleitos_dados_demograficos[df_eleitos_dados_demograficos['DS_CARGO'] == 'PREFEITO']
df_eleitos_vereadores = df_eleitos_dados_demograficos[df_eleitos_dados_demograficos['DS_CARGO'] == 'VEREADOR']


# Exibir o tamanho dos DataFrames
print("Tamanho de df_eleitos_prefeitos:", df_eleitos_prefeitos.shape)

print("Tamanho de df_eleitos_vereadores:", df_eleitos_vereadores.shape)

# Exibir informações sobre os DataFrames
print("\nInformações sobre df_prefeitos:")
print(df_eleitos_prefeitos.info())
print("\nInformações sobre df_vereadores:")
print(df_eleitos_vereadores.info())


# Amostra de 5 registros aleatórios de cada DataFrame
print("\nAmostra de 5 registros de df_prefeitos:")
print(df_eleitos_prefeitos.sample(5))
print("\nAmostra de 5 registros de df_vereadores:")
print(df_eleitos_vereadores.sample(5))


#Variáveis Categóricas:
# Identificar e imprimir as variáveis categóricas
categorical_columns = df_eleitos_dados_demograficos.select_dtypes(include='category').columns
print("Variáveis Categóricas: ")
for col in categorical_columns:
    print(f"Coluna: {col}")
    print(df_eleitos_dados_demograficos[col].value_counts())
    print("\n")




# Realizar análise ANOVA nas colunas categóricas
print("Análise ANOVA para df_eleitos_dados_demograficos (colunas categóricas)")
analisar_ANOVA(df_eleitos_dados_demograficos,colunas_categoricas)

print("Análise ANOVA para df_prefeitos (colunas categóricas)")
analisar_ANOVA(df_eleitos_prefeitos,colunas_categoricas)

print("Análise ANOVA para df_vereadores (colunas categóricas)")
analisar_ANOVA(df_eleitos_vereadores,colunas_categoricas)


#Variáveis Numéricas:
# Identificar e imprimir as variáveis numéricas
print("Variáveis Numéricas :")
# Exibir um resumo estatístico das colunas numéricas do DataFrame
print("Resumo estatístico das colunas numéricas de df_eleitos_dados_demograficos:")
print(df_eleitos_dados_demograficos.describe())

# Plotar matrizes de correlação para os DataFrames
plotar_matriz_correlacao(df_eleitos_dados_demograficos,colunas_numericas,"df_eleitos_dados_demograficos")
plotar_matriz_correlacao(df_eleitos_prefeitos,colunas_numericas,"df_eleitos_prefeitos")
plotar_matriz_correlacao(df_eleitos_vereadores,colunas_numericas,"df_eleitos_vereadores")

# Plotar boxplots para os DataFrames
plot_boxplots(df_eleitos_dados_demograficos,colunas_numericas,"df_eleitos_dados_demograficos")
plot_boxplots(df_eleitos_prefeitos,colunas_numericas,"df_eleitos_prefeitos")
plot_boxplots(df_eleitos_vereadores,colunas_numericas,"df_eleitos_vereadores")

# Plotar boxplot específico para a coluna 'VR_DESPESA_MAX_CAMPANHA'
plot_boxplot(df_eleitos_dados_demograficos,"df_eleitos_dados_demograficos",'VR_DESPESA_MAX_CAMPANHA')
plot_boxplot(df_eleitos_prefeitos, "df_eleitos_prefeitos" ,'VR_DESPESA_MAX_CAMPANHA')
plot_boxplot(df_eleitos_vereadores,"df_eleitos_vereadores" ,'VR_DESPESA_MAX_CAMPANHA')

# Imprimir estatísticas descritivas para a coluna 'VR_DESPESA_MAX_CAMPANHA' em cada DataFrame
print("Estatísticas para VR_DESPESA_MAX_CAMPANHA em df_eleitos_dados_demograficos:")
print(df_eleitos_dados_demograficos['VR_DESPESA_MAX_CAMPANHA'].describe())
print("Estatísticas para VR_DESPESA_MAX_CAMPANHA em df_prefeitos:")
print(df_eleitos_prefeitos['VR_DESPESA_MAX_CAMPANHA'].describe())
print("Estatísticas para VR_DESPESA_MAX_CAMPANHA em df_vereadores:")
print(df_eleitos_vereadores['VR_DESPESA_MAX_CAMPANHA'].describe())


print('*' * 100)
print('INICIO analise_de_modelo_VR_DESPESA_MAX_CAMPANHA')
print('*' * 100)

#df_eleitos_dados_demograficos

# Análise do modelo de regressão para df_eleitos_dados_demograficos VR_DESPESA_MAX_CAMPANHA com outliers e com análise de resíduos
print("Análise do modelo de regressão para df_eleitos_dados_demograficos VR_DESPESA_MAX_CAMPANHA com outliers:")
analise_de_modelo_VR_DESPESA_MAX_CAMPANHA(df_eleitos_dados_demograficos,"df_eleitos_dados_demograficos com outliers","S")

#Remover Outliers de df_eleitos_dados_demograficos e criar o dataframe df_eleitos_dados_demograficos_outliers contendo apenas o outilers correpondentes
df_eleitos_dados_demograficos, df_eleitos_dados_demograficos_outliers = remover_outliers_VR_DESPESA_MAX_CAMPANHA(df_eleitos_dados_demograficos,"df_eleitos_dados_demograficos com outliers")

# Análise do modelo de regressão para df_eleitos_dados_demograficos VR_DESPESA_MAX_CAMPANHA sem outliers e sem análise de resíduos
print("Análise do modelo de regressão para df_eleitos_dados_demograficos VR_DESPESA_MAX_CAMPANHA sem outliers")
analise_de_modelo_VR_DESPESA_MAX_CAMPANHA(df_eleitos_dados_demograficos,"df_eleitos_dados_demograficos sem outliers","N")


plot_boxplots(df_eleitos_dados_demograficos,colunas_numericas,"df_eleitos_dados_demograficos sem outliers")
plot_boxplot(df_eleitos_dados_demograficos,"df_eleitos_dados_demograficos sem outliers",'VR_DESPESA_MAX_CAMPANHA')



#df_eleitos_prefeitos

print("Análise do modelo de regressão para df_eleitos_prefeitos VR_DESPESA_MAX_CAMPANHA com outliers:")
# Análise do modelo de regressão para df_eleitos_prefeitos VR_DESPESA_MAX_CAMPANHA com outliers e com análise de resíduos
analise_de_modelo_VR_DESPESA_MAX_CAMPANHA(df_eleitos_prefeitos,"df_eleitos_prefeitos com outliers","S")

#Remover Outliers de df_eleitos_prefeitos e criar o dataframe df_eleitos_prefeitos_outliers contendo apenas o outilers correpondentes
df_eleitos_prefeitos, df_eleitos_prefeitos_outliers = remover_outliers_VR_DESPESA_MAX_CAMPANHA(df_eleitos_prefeitos,"df_eleitos_prefeitos com outliers")

# Análise do modelo de regressão para df_eleitos_prefeitos VR_DESPESA_MAX_CAMPANHA sem outliers e sem análise de resíduos
print("Análise do modelo de regressão para df_eleitos_prefeitos VR_DESPESA_MAX_CAMPANHA sem outliers:")
analise_de_modelo_VR_DESPESA_MAX_CAMPANHA(df_eleitos_prefeitos,"df_eleitos_prefeitos sem outliers","N")


plot_boxplots(df_eleitos_prefeitos,colunas_numericas ,'df_eleitos_prefeitos sem outliers')
plot_boxplot(df_eleitos_prefeitos, "df_eleitos_prefeitos sem outliers" ,'VR_DESPESA_MAX_CAMPANHA')




#df_eleitos_vereadores
# Análise do modelo de regressão para df_eleitos_vereadores VR_DESPESA_MAX_CAMPANHA com outliers e com análise de resíduos
print("Análise do modelo de regressão para df_eleitos_vereadores VR_DESPESA_MAX_CAMPANHA com outliers:")
analise_de_modelo_VR_DESPESA_MAX_CAMPANHA(df_eleitos_vereadores,"df_eleitos_vereadores com outliers","S")

#Remover Outliers de df_eleitos_vereadores e criar o dataframe df_eleitos_vereadores_outliers contendo apenas o outilers correpondentes
df_eleitos_vereadores, df_eleitos_vereadores_outliers = remover_outliers_VR_DESPESA_MAX_CAMPANHA(df_eleitos_vereadores,"df_eleitos_vereadores com outliers")

# Análise do modelo de regressão para df_eleitos_vereadores VR_DESPESA_MAX_CAMPANHA sem outliers e sem análise de resíduos
print("Análise do modelo de regressão para df_eleitos_vereadores VR_DESPESA_MAX_CAMPANHA sem outliers:")
analise_de_modelo_VR_DESPESA_MAX_CAMPANHA(df_eleitos_vereadores,"df_eleitos_vereadores sem outliers","N")

plot_boxplots(df_eleitos_vereadores,colunas_numericas,'df_eleitos_vereadores sem outliers')
plot_boxplot(df_eleitos_vereadores,"df_eleitos_vereadores" ,'VR_DESPESA_MAX_CAMPANHA')


print('*' * 100)
print('FIM analise_de_modelo_VR_DESPESA_MAX_CAMPANHA')
print('*' * 100)

print('*' * 100)
print('INICIO Análise Outliers')
print('*' * 100)

# Calcula a média de 'VR_DESPESA_MAX_CAMPANHA' por município
agrupado_por_municipio_vereadores = df_eleitos_vereadores_outliers.groupby('NOME DO MUNICÍPIO')['VR_DESPESA_MAX_CAMPANHA'].mean().reset_index().copy()

# Ordena o DataFrame resultante em ordem decrescente com base na média de despesas de campanha
agrupado_por_municipio_vereadores = agrupado_por_municipio_vereadores.sort_values(by='VR_DESPESA_MAX_CAMPANHA', ascending=False)

# Calcula a média de 'VR_DESPESA_MAX_CAMPANHA' por município, distinguindo entre capitais e não capitais
agrupado_por_municipio_vereadores_capital = df_eleitos_vereadores_outliers.groupby(['capital', 'NOME DO MUNICÍPIO'])['VR_DESPESA_MAX_CAMPANHA'].mean().reset_index()

# Calcula a média de 'VR_DESPESA_MAX_CAMPANHA' em capitais
agrupado_por_municipio_vereadores_capital_1 = agrupado_por_municipio_vereadores_capital[
    (agrupado_por_municipio_vereadores_capital['capital'] == 1)
    & agrupado_por_municipio_vereadores_capital['VR_DESPESA_MAX_CAMPANHA'].notnull()
].groupby(['capital', 'NOME DO MUNICÍPIO'])['VR_DESPESA_MAX_CAMPANHA'].mean().reset_index()

agrupado_por_municipio_vereadores_capital['Média Capital'] =  agrupado_por_municipio_vereadores_capital_1['VR_DESPESA_MAX_CAMPANHA'].mean()




# Calcula a média de 'VR_DESPESA_MAX_CAMPANHA' em não capitais
agrupado_por_municipio_vereadores_capital_0 = agrupado_por_municipio_vereadores_capital[
    (agrupado_por_municipio_vereadores_capital['capital'] == 0)
    & agrupado_por_municipio_vereadores_capital['VR_DESPESA_MAX_CAMPANHA'].notnull()
].groupby(['capital', 'NOME DO MUNICÍPIO'])['VR_DESPESA_MAX_CAMPANHA'].mean().reset_index()


agrupado_por_municipio_vereadores_capital['Média Não Capital'] =  agrupado_por_municipio_vereadores_capital_0['VR_DESPESA_MAX_CAMPANHA'].mean()


# Calcula e exibe a média geral de 'VR_DESPESA_MAX_CAMPANHA' por município
print('Média de VR_DESPESA_MAX_CAMPANHA por Município - df_eleitos_vereadores_outliers:')
print(agrupado_por_municipio_vereadores)

# Calcula e exibe a média de 'VR_DESPESA_MAX_CAMPANHA' em capitais
print("VR_DESPESA_MAX_CAMPANHA médio em capitais - df_eleitos_vereadores_outliers")
print(agrupado_por_municipio_vereadores_capital_1['VR_DESPESA_MAX_CAMPANHA'].mean())

# Calcula e exibe a média de 'VR_DESPESA_MAX_CAMPANHA' em não capitais
print("VR_DESPESA_MAX_CAMPANHA médio em não capitais - df_eleitos_vereadores_outliers")
print(agrupado_por_municipio_vereadores_capital_0['VR_DESPESA_MAX_CAMPANHA'].mean())


# Agrupa os dados por 'NOME DO MUNICÍPIO' e calcula a média da despesa máxima de campanha (VR_DESPESA_MAX_CAMPANHA)
# para prefeitos eleitos, removendo outliers e reiniciando o índice.
agrupado_por_municipio_prefeitos = df_eleitos_prefeitos_outliers.groupby('NOME DO MUNICÍPIO')['VR_DESPESA_MAX_CAMPANHA'].mean().reset_index().copy()

# Ordena o DataFrame resultante em ordem decrescente com base na média da despesa máxima de campanha.
agrupado_por_municipio_prefeitos = agrupado_por_municipio_prefeitos.sort_values(by='VR_DESPESA_MAX_CAMPANHA', ascending=False)

# Calcula a média da despesa máxima de campanha por município, distinguindo entre capitais e não capitais.
agrupado_por_municipio_prefeitos_capital = df_eleitos_prefeitos_outliers.groupby(['capital', 'NOME DO MUNICÍPIO'])['VR_DESPESA_MAX_CAMPANHA'].mean().reset_index()

# Calcula a média da despesa máxima de campanha para capitais.
agrupado_por_municipio_prefeitos_capital_1 = agrupado_por_municipio_prefeitos_capital[
    (agrupado_por_municipio_prefeitos_capital['capital'] == 1) &
    (agrupado_por_municipio_prefeitos_capital['VR_DESPESA_MAX_CAMPANHA'].notnull())
].groupby(['capital', 'NOME DO MUNICÍPIO'])['VR_DESPESA_MAX_CAMPANHA'].mean().reset_index()

# Adiciona a média da despesa máxima de campanha para capitais ao DataFrame principal.
agrupado_por_municipio_prefeitos_capital['Média Capital'] = agrupado_por_municipio_prefeitos_capital_1['VR_DESPESA_MAX_CAMPANHA'].mean()

# Calcula a média da despesa máxima de campanha para não capitais.
agrupado_por_municipio_prefeitos_capital_0 = agrupado_por_municipio_prefeitos_capital[
    (agrupado_por_municipio_prefeitos_capital['capital'] == 0) &
    (agrupado_por_municipio_prefeitos_capital['VR_DESPESA_MAX_CAMPANHA'].notnull())
].groupby(['capital', 'NOME DO MUNICÍPIO'])['VR_DESPESA_MAX_CAMPANHA'].mean().reset_index()

# Adiciona a média da despesa máxima de campanha para não capitais ao DataFrame principal.
agrupado_por_municipio_prefeitos_capital['Média Não Capital'] = agrupado_por_municipio_prefeitos_capital_0['VR_DESPESA_MAX_CAMPANHA'].mean()

# Imprime a média da despesa máxima de campanha por município.
print('VR_DESPESA_MAX_CAMPANHA por Município - df_eleitos_prefeitos_outliers:')
print(agrupado_por_municipio_prefeitos)

# Imprime a média da despesa máxima de campanha em capitais e não capitais.
print("Média da VR_DESPESA_MAX_CAMPANHA em Capitais - df_eleitos_prefeitos_outliers")
print(agrupado_por_municipio_prefeitos_capital_1['VR_DESPESA_MAX_CAMPANHA'].mean())
print("Média da VR_DESPESA_MAX_CAMPANHA em Não Capitais - df_eleitos_prefeitos_outliers")
print(agrupado_por_municipio_prefeitos_capital_0['VR_DESPESA_MAX_CAMPANHA'].mean())

# Define a paleta de cores para os gráficos usando o esquema "viridis" com 30 cores diferentes.
sns.set_palette("viridis", 30)

# Cria uma figura com dois subplots de tamanho personalizado (16 unidades de largura por 6 unidades de altura).
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Primeiro gráfico de barras (por UF)
# Utiliza um gráfico de barras para representar os gastos máximos de campanha por município.
# Os municípios são ordenados pela ordem original.
sns.barplot(data=agrupado_por_municipio_vereadores, x='NOME DO MUNICÍPIO', y='VR_DESPESA_MAX_CAMPANHA', ax=axes[0],
            order=agrupado_por_municipio_vereadores['NOME DO MUNICÍPIO'])

# Adiciona um gráfico de linha representando a média dos gastos máximos de campanha em capitais.
sns.lineplot(data=agrupado_por_municipio_vereadores_capital, x='NOME DO MUNICÍPIO', y='Média Capital',
             label='VR_DESPESA_MAX_CAMPANHA médio em capitais', linestyle='dashed', color='Red', ax=axes[0])

# Adiciona um gráfico de linha representando a média dos gastos máximos de campanha em não capitais.
sns.lineplot(data=agrupado_por_municipio_vereadores_capital, x='NOME DO MUNICÍPIO', y='Média Não Capital',
             label='VR_DESPESA_MAX_CAMPANHA médio  em não capitais', linestyle='dashed', color='Blue', ax=axes[0])

# Define o título, rótulos dos eixos e formatação dos ticks para o primeiro gráfico.
axes[0].set_title('Média de VR_DESPESA_MAX_CAMPANHA por Município - df_eleitos_vereadores_outliers')
axes[0].set_xlabel('Município')
axes[0].set_ylabel('Média VR_DESPESA_MAX_CAMPANHA')
axes[0].tick_params(axis='x', rotation=90)
axes[0].ticklabel_format(axis='y', style='plain', useOffset=False)

# Define uma nova paleta de cores para o segundo gráfico com 50 cores diferentes.
sns.set_palette("viridis", 50)

# Segundo gráfico de barras (por Região)
# Utiliza um gráfico de barras para representar os gastos máximos de campanha por município.
# Os municípios são ordenados pela ordem original.
sns.barplot(data=agrupado_por_municipio_prefeitos, x='NOME DO MUNICÍPIO', y='VR_DESPESA_MAX_CAMPANHA', ax=axes[1],
            order=agrupado_por_municipio_prefeitos['NOME DO MUNICÍPIO'])

# Adiciona um gráfico de linha representando a média dos gastos máximos de campanha em capitais.
sns.lineplot(data=agrupado_por_municipio_prefeitos_capital, x='NOME DO MUNICÍPIO', y='Média Capital',
             label='VR_DESPESA_MAX_CAMPANHA médio em capitais', linestyle='dashed', color='Red', ax=axes[1])

# Adiciona um gráfico de linha representando a média dos gastos máximos de campanha em não capitais.
sns.lineplot(data=agrupado_por_municipio_prefeitos_capital, x='NOME DO MUNICÍPIO', y='Média Não Capital',
             label='VR_DESPESA_MAX_CAMPANHA médio em não capitais', linestyle='dashed', color='Blue', ax=axes[1])

# Define o título, rótulos dos eixos e formatação dos ticks para o segundo gráfico.
axes[1].set_title('VR_DESPESA_MAX_CAMPANHA por Município - df_eleitos_prefeitos_outliers')
axes[1].set_xlabel('Município')
axes[1].set_ylabel('VR_DESPESA_MAX_CAMPANHA')
axes[1].tick_params(axis='x', rotation=90)
axes[1].ticklabel_format(axis='y', style='plain', useOffset=False)

# Ajusta o layout e exibe os gráficos.
plt.tight_layout()
plt.show()


# Configurando a paleta de cores para o gráfico
sns.set_palette("viridis", 30)

# Agrupando os dados de vereadores por unidade federativa (UF) e calculando a média das despesas máximas de campanha e PIB per capita
agrupado_por_uf_pib_vereadores = df_eleitos_vereadores_outliers.groupby('UF')[['VR_DESPESA_MAX_CAMPANHA', 'Produto Interno Bruto per capita, a preços correntes (R$ 1,00)']].mean().reset_index()

# Classificando os dados por despesas máximas de campanha de vereadores em ordem decrescente
agrupado_por_uf_pib_vereadores = agrupado_por_uf_pib_vereadores.sort_values(by=['VR_DESPESA_MAX_CAMPANHA'], ascending=False)

# Classificando os dados por PIB per capita em ordem decrescente
agrupado_por_uf_pib_vereadores_ord = agrupado_por_uf_pib_vereadores.sort_values(by=['Produto Interno Bruto per capita, a preços correntes (R$ 1,00)'], ascending=False)

# Imprimindo a média das despesas máximas de campanha por UF para vereadores
print('Média de VR_DESPESA_MAX_CAMPANHA por UF - df_eleitos_vereadores_outliers:')
print(agrupado_por_uf_pib_vereadores)

# Imprimindo a média do PIB per capita por UF para vereadores
print('Média de PIB, a preços correntes (R$ 1,00) por UF - df_eleitos_vereadores_outliers:')
print(agrupado_por_uf_pib_vereadores_ord)

# Agrupando os dados de prefeitos por unidade federativa (UF) e calculando a média das despesas máximas de campanha e PIB per capita
agrupado_por_uf_pib_prefeitos = df_eleitos_prefeitos_outliers.groupby('UF')[['VR_DESPESA_MAX_CAMPANHA', 'Produto Interno Bruto per capita, a preços correntes (R$ 1,00)']].mean().reset_index()

# Classificando os dados por despesas máximas de campanha de prefeitos em ordem decrescente
agrupado_por_uf_pib_prefeitos = agrupado_por_uf_pib_prefeitos.sort_values(by=['VR_DESPESA_MAX_CAMPANHA'], ascending=False)

# Classificando os dados por PIB per capita em ordem decrescente
agrupado_por_uf_pib_prefeitos_ord = agrupado_por_uf_pib_prefeitos.sort_values(by=['Produto Interno Bruto per capita, a preços correntes (R$ 1,00)'], ascending=False)

# Imprimindo a média das despesas máximas de campanha por UF para prefeitos
print('Média de VR_DESPESA_MAX_CAMPANHA por UF - df_eleitos_prefeitos_outliers:')
print(agrupado_por_uf_pib_prefeitos)

# Imprimindo a média do PIB per capita por UF para prefeitos
print('Média de PIB per capita, a preços correntes (R$ 1,00) por UF - df_eleitos_prefeitos_outliers:')
print(agrupado_por_uf_pib_prefeitos_ord)


# Cria uma figura com dois subplots em duas linhas e duas colunas
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Primeiro gráfico de barras (vereadores) - Média do VR_DESPESA_MAX_CAMPANHA por UF
sns.barplot(data=agrupado_por_uf_pib_vereadores, x='UF', y='VR_DESPESA_MAX_CAMPANHA', ax=axes[0, 0], order=agrupado_por_uf_pib_vereadores['UF'])
axes[0, 0].set_title('Média de VR_DESPESA_MAX_CAMPANHA por UF - df_eleitos_vereadores_outliers')
axes[0, 0].set_xlabel('UF')
axes[0, 0].set_ylabel('Média de Gastos Máximos de Campanha')
axes[0, 0].tick_params(axis='x', rotation=90)
axes[0, 0].ticklabel_format(axis='y', style='plain', useOffset=False)

# Segundo gráfico de barras (vereadores) - Média do PIB per capita por UF
sns.barplot(data=agrupado_por_uf_pib_vereadores_ord, x='UF', y='Produto Interno Bruto per capita, a preços correntes (R$ 1,00)', ax=axes[0, 1], order=agrupado_por_uf_pib_vereadores_ord['UF'])
axes[0, 1].set_title('Média do PIB per capita por UF - df_eleitos_vereadores_outliers')
axes[0, 1].set_xlabel('UF')
axes[0, 1].set_ylabel('Média do PIB per capita, a preços correntes (R$ 1,00)')
axes[0, 1].tick_params(axis='x', rotation=90)
axes[0, 1].ticklabel_format(axis='y', style='plain', useOffset=False)

# Terceiro gráfico de barras (prefeitos) - Média do VR_DESPESA_MAX_CAMPANHA por UF
sns.barplot(data=agrupado_por_uf_pib_prefeitos, x='UF', y='VR_DESPESA_MAX_CAMPANHA', ax=axes[1, 0], order=agrupado_por_uf_pib_prefeitos['UF'])
axes[1, 0].set_title('Média de VR_DESPESA_MAX_CAMPANHA por UF - df_eleitos_prefeitos_outliers')
axes[1, 0].set_xlabel('UF')
axes[1, 0].set_ylabel('Média de VR_DESPESA_MAX_CAMPANHA')
axes[1, 0].tick_params(axis='x', rotation=90)
axes[1, 0].ticklabel_format(axis='y', style='plain', useOffset=False)

# Quarto gráfico de barras (prefeitos) - Média do PIB per capita por UF
sns.barplot(data=agrupado_por_uf_pib_prefeitos_ord, x='UF', y='Produto Interno Bruto per capita, a preços correntes (R$ 1,00)', ax=axes[1, 1], order=agrupado_por_uf_pib_prefeitos_ord['UF'])
axes[1, 1].set_title('Média do PIB per capita por UF - df_eleitos_prefeitos_outliers')
axes[1, 1].set_xlabel('UF')
axes[1, 1].set_ylabel('Média do PIB per capita, a preços correntes (R$ 1,00)')
axes[1, 1].tick_params(axis='x', rotation=90)
axes[1, 1].ticklabel_format(axis='y', style='plain', useOffset=False)

# Ajuste de layout para evitar sobreposição
plt.tight_layout()

# Exiba os gráficos
plt.show()


# Para o primeiro gráfico (vereadores):

# Agrupa os dados por 'UF' e calcula a média das colunas 'VR_DESPESA_MAX_CAMPANHA' e 'POPULAÇÃO ESTIMADA'.
agrupado_por_uf_pop_vereadores = df_eleitos_vereadores_outliers.groupby('UF')[['VR_DESPESA_MAX_CAMPANHA', 'POPULAÇÃO ESTIMADA']].mean().reset_index()

# Ordena o DataFrame pela média de 'VR_DESPESA_MAX_CAMPANHA' em ordem decrescente.
agrupado_por_uf_pop_vereadores = agrupado_por_uf_pop_vereadores.sort_values(by=['VR_DESPESA_MAX_CAMPANHA'], ascending=False)

# Ordena o DataFrame pela média de 'POPULAÇÃO ESTIMADA' em ordem decrescente.
agrupado_por_uf_pop_vereadores_ord = agrupado_por_uf_pop_vereadores.sort_values(by=['POPULAÇÃO ESTIMADA'], ascending=False)

# Exibe as médias de gastos de campanha para vereadores por UF.
print('Média de VR_DESPESA_MAX_CAMPANHA por UF - df_eleitos_vereadores_outliers:')
print(agrupado_por_uf_pop_vereadores)

# Exibe as médias de população estimada por UF, ordenadas por gastos de campanha para vereadores.
print('Média de POPULAÇÃO ESTIMADA por UF - df_eleitos_vereadores_outliers:')
print(agrupado_por_uf_pop_vereadores_ord)


# Para o segundo gráfico (prefeitos):

# Agrupa os dados por 'UF' e calcula a média das colunas 'VR_DESPESA_MAX_CAMPANHA' e 'POPULAÇÃO ESTIMADA'.
agrupado_por_uf_pop_prefeitos = df_eleitos_prefeitos_outliers.groupby('UF')[['VR_DESPESA_MAX_CAMPANHA', 'POPULAÇÃO ESTIMADA']].mean().reset_index()

# Ordena o DataFrame pela média de 'VR_DESPESA_MAX_CAMPANHA' em ordem decrescente.
agrupado_por_uf_pop_prefeitos = agrupado_por_uf_pop_prefeitos.sort_values(by=['VR_DESPESA_MAX_CAMPANHA'], ascending=False)

# Ordena o DataFrame pela média de 'POPULAÇÃO ESTIMADA' em ordem decrescente.
agrupado_por_uf_pop_prefeitos_ord = agrupado_por_uf_pop_prefeitos.sort_values(by=['POPULAÇÃO ESTIMADA'], ascending=False)

# Exibe as médias de gastos de campanha para prefeitos por UF.
print('Média de VR_DESPESA_MAX_CAMPANHA por UF - df_eleitos_prefeitos_outliers:')
print(agrupado_por_uf_pop_prefeitos)

# Exibe as médias de população estimada por UF, ordenadas por gastos de campanha para prefeitos.
print('Média de POPULAÇÃO ESTIMADA por UF - df_eleitos_prefeitos_outliers:')
print(agrupado_por_uf_pop_prefeitos_ord)


# Cria uma figura com dois subplots em duas linhas e duas colunas
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Primeiro gráfico de barras (vereadores) - Média VR_DESPESA_MAX_CAMPANHA por UF
sns.barplot(data=agrupado_por_uf_pop_vereadores, x='UF', y='VR_DESPESA_MAX_CAMPANHA', ax=axes[0, 0], order=agrupado_por_uf_pop_vereadores['UF'])
axes[0, 0].set_title('Média de VR_DESPESA_MAX_CAMPANHA por UF - df_eleitos_vereadores_outliers')
axes[0, 0].set_xlabel('UF')
axes[0, 0].set_ylabel('Média de Gastos Máximos de Campanha')
axes[0, 0].tick_params(axis='x', rotation=90)
axes[0, 0].ticklabel_format(axis='y', style='plain', useOffset=False)

# Segundo gráfico de barras (vereadores) - Média POPULAÇÃO ESTIMADA por UF
sns.barplot(data=agrupado_por_uf_pop_vereadores_ord, x='UF', y='POPULAÇÃO ESTIMADA', ax=axes[0, 1], order=agrupado_por_uf_pop_vereadores_ord['UF'])
axes[0, 1].set_title('Média de POPULAÇÃO ESTIMADA por UF - df_eleitos_vereadores_outliers')
axes[0, 1].set_xlabel('UF')
axes[0, 1].set_ylabel('Média de POPULAÇÃO ESTIMADA ')
axes[0, 1].tick_params(axis='x', rotation=90)
axes[0, 1].ticklabel_format(axis='y', style='plain', useOffset=False)

# Terceiro gráfico de barras (prefeitos) - Média VR_DESPESA_MAX_CAMPANHA por UF
sns.barplot(data=agrupado_por_uf_pop_prefeitos, x='UF', y='VR_DESPESA_MAX_CAMPANHA', ax=axes[1, 0], order=agrupado_por_uf_pop_prefeitos['UF'])
axes[1, 0].set_title('Média de VR_DESPESA_MAX_CAMPANHA por UF - df_eleitos_prefeitos_outliers')
axes[1, 0].set_xlabel('UF')
axes[1, 0].set_ylabel('Média de VR_DESPESA_MAX_CAMPANHA')
axes[1, 0].tick_params(axis='x', rotation=90)
axes[1, 0].ticklabel_format(axis='y', style='plain', useOffset=False)

# Quarto gráfico de barras (prefeitos) - Média POPULAÇÃO ESTIMADA por UF
sns.barplot(data=agrupado_por_uf_pop_prefeitos_ord, x='UF', y='POPULAÇÃO ESTIMADA', ax=axes[1, 1], order=agrupado_por_uf_pop_prefeitos_ord['UF'])
axes[1, 1].set_title('Média de POPULAÇÃO ESTIMADA por UF - df_eleitos_prefeitos_outliers')
axes[1, 1].set_xlabel('UF')
axes[1, 1].set_ylabel('Média de POPULAÇÃO ESTIMADA')
axes[1, 1].tick_params(axis='x', rotation=90)
axes[1, 1].ticklabel_format(axis='y', style='plain', useOffset=False)

# Ajusta o layout para evitar sobreposição de elementos
plt.tight_layout()

# Exibe os gráficos
plt.show()


# Agrupamento de dados por 'UF' e cálculo da média de 'VR_DESPESA_MAX_CAMPANHA' para o primeiro gráfico
agrupado_por_uf = df_eleitos_prefeitos_outliers.groupby('UF')['VR_DESPESA_MAX_CAMPANHA'].mean().reset_index()
# Ordenação do DataFrame resultante em ordem decrescente pela média de despesas máximas de campanha
agrupado_por_uf = agrupado_por_uf.sort_values(by='VR_DESPESA_MAX_CAMPANHA', ascending=False)

# Exibição da média de 'VR_DESPESA_MAX_CAMPANHA' por UF - DataFrame: df_eleitos_prefeitos_outliers
print("Média de VR_DESPESA_MAX_CAMPANHA por UF - df_eleitos_prefeitos_outliers")
print(agrupado_por_uf)

# Agrupamento de dados por 'Nome_da_Grande_Região' e cálculo da média de 'VR_DESPESA_MAX_CAMPANHA' para o segundo gráfico
agrupado_por_regiao = df_eleitos_prefeitos_outliers.groupby('Nome_da_Grande_Região')['VR_DESPESA_MAX_CAMPANHA'].mean().reset_index()
# Ordenação do DataFrame resultante em ordem decrescente pela média de despesas máximas de campanha
agrupado_por_regiao = agrupado_por_regiao.sort_values(by='VR_DESPESA_MAX_CAMPANHA', ascending=False)

# Exibição da média de 'VR_DESPESA_MAX_CAMPANHA' por Região - DataFrame: df_eleitos_prefeitos_outliers
print("Média de VR_DESPESA_MAX_CAMPANHA por Região - df_eleitos_prefeitos_outliers")
print(agrupado_por_regiao)

# Criação de uma figura com dois subplots para os gráficos
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Primeiro gráfico de barras (por UF)
sns.barplot(data=agrupado_por_uf, x='UF', y='VR_DESPESA_MAX_CAMPANHA', ax=axes[0], order=agrupado_por_uf['UF'])
axes[0].set_title('Média de VR_DESPESA_MAX_CAMPANHA por UF - df_eleitos_prefeitos_outliers')
axes[0].set_xlabel('UF')
axes[0].set_ylabel('Média VR_DESPESA_MAX_CAMPANHA')
axes[0].tick_params(axis='x', rotation=90)
axes[0].ticklabel_format(axis='y', style='plain', useOffset=False)

# Segundo gráfico de barras (por Região)
sns.barplot(data=agrupado_por_regiao, x='Nome_da_Grande_Região', y='VR_DESPESA_MAX_CAMPANHA', ax=axes[1], order=agrupado_por_regiao['Nome_da_Grande_Região'])
axes[1].set_title('Média de VR_DESPESA_MAX_CAMPANHA por Região - df_eleitos_prefeitos_outliers')
axes[1].set_xlabel('Região')
axes[1].set_ylabel('Média VR_DESPESA_MAX_CAMPANHA')
axes[1].tick_params(axis='x', rotation=90)
axes[1].ticklabel_format(axis='y', style='plain', useOffset=False)

# Ajuste de layout e exibição dos gráficos
plt.tight_layout()
plt.show()


# Realiza o agrupamento e ordenação dos dados para o primeiro gráfico de barras (vereadores)
agrupado_por_partido_vereadores = df_eleitos_vereadores_outliers.groupby('SG_PARTIDO')['VR_DESPESA_MAX_CAMPANHA'].mean().reset_index()
agrupado_por_partido_vereadores = agrupado_por_partido_vereadores.sort_values(by='VR_DESPESA_MAX_CAMPANHA', ascending=False)

# Imprime a média de VR_DESPESA_MAX_CAMPANHA por Partido para vereadores
print("Média de VR_DESPESA_MAX_CAMPANHA por Partido - df_eleitos_vereadores_outliers")
print(agrupado_por_partido_vereadores)

# Realiza o agrupamento e ordenação dos dados para o segundo gráfico de barras (prefeitos)
agrupado_por_partido_prefeitos = df_eleitos_prefeitos_outliers.groupby('SG_PARTIDO')['VR_DESPESA_MAX_CAMPANHA'].mean().reset_index()
agrupado_por_partido_prefeitos = agrupado_por_partido_prefeitos.sort_values(by='VR_DESPESA_MAX_CAMPANHA', ascending=False)

# Imprime a média de VR_DESPESA_MAX_CAMPANHA por Partido para prefeitos
print("Média de VR_DESPESA_MAX_CAMPANHA por Partido - df_eleitos_prefeitos_outliers")
print(agrupado_por_partido_prefeitos)

# Cria uma figura com dois subgráficos (um para cada gráfico de barras)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Configura o primeiro gráfico de barras (vereadores)
sns.barplot(data=agrupado_por_partido_vereadores, x='SG_PARTIDO', y='VR_DESPESA_MAX_CAMPANHA', ax=axes[0],
            order=agrupado_por_partido_vereadores['SG_PARTIDO'])
axes[0].set_title('Média de VR_DESPESA_MAX_CAMPANHA por Partido - df_eleitos_vereadores_outliers')
axes[0].set_xlabel('Partido')
axes[0].set_ylabel('Média VR_DESPESA_MAX_CAMPANHA')
axes[0].tick_params(axis='x', rotation=90)
axes[0].ticklabel_format(axis='y', style='plain', useOffset=False)

# Configura o segundo gráfico de barras (prefeitos)
sns.barplot(data=agrupado_por_partido_prefeitos, x='SG_PARTIDO', y='VR_DESPESA_MAX_CAMPANHA', ax=axes[1],
            order=agrupado_por_partido_prefeitos['SG_PARTIDO'])
axes[1].set_title('Média de VR_DESPESA_MAX_CAMPANHA por Partido - df_eleitos_prefeitos_outliers')
axes[1].set_xlabel('Partido')
axes[1].set_ylabel('Média VR_DESPESA_MAX_CAMPANHA')
axes[1].tick_params(axis='x', rotation=90)
axes[1].ticklabel_format(axis='y', style='plain', useOffset=False)

# Ajusta o layout para garantir que os gráficos não se sobreponham
plt.tight_layout()

# Exibe a figura com os dois gráficos de barras
plt.show()

# Define a paleta de cores a ser usada nos gráficos
sns.set_palette("Paired", 30)

# Cria uma figura com três subgráficos
fig, axes = plt.subplots(1, 3, figsize=(18, 6))


# Agrupa os dados por 'ST_REELEICAO' para diferentes conjuntos de dados
agrupado_por_reeleicao_demograficos = df_eleitos_dados_demograficos_outliers.groupby(
    'ST_REELEICAO').count().reset_index()
agrupado_por_reeleicao_prefeitos = df_eleitos_prefeitos_outliers.groupby('ST_REELEICAO').count().reset_index()
agrupado_por_reeleicao_vereadores = df_eleitos_vereadores_outliers.groupby('ST_REELEICAO').count().reset_index()

# Plota os gráficos de donut para diferentes conjuntos de dados
plot_donut(agrupado_por_reeleicao_demograficos,
           'Percentual Situação Reeleição - df_eleitos_dados_demograficos_outliers', axes[0])
plot_donut(agrupado_por_reeleicao_prefeitos, 'Percentual Situação Reeleição - df_eleitos_prefeitos_outliers', axes[1])
plot_donut(agrupado_por_reeleicao_vereadores, 'Percentual Situação Reeleição - df_eleitos_vereadores_outliers', axes[2])

# Imprime informações sobre os dados agrupados
print('Percentual Situação Reeleição - df_eleitos_dados_demograficos_outliers')
print(agrupado_por_reeleicao_demograficos)

print('Percentual Situação Reeleição - df_eleitos_prefeitos_outliers')
print(agrupado_por_reeleicao_prefeitos)

print('Percentual Situação Reeleição - df_eleitos_vereadores_outliers')
print(agrupado_por_reeleicao_vereadores)

# Ajusta o layout para evitar sobreposições
plt.tight_layout()

# Exibe o gráfico de donut contendo os três gráficos de pizza
plt.show()

#Agrupa os dados por 'DS_CARGO' e calcula a contagem de ocorrências (eleitos) para o primeiro gráfico
agrupado_por_cargo = df_eleitos_dados_demograficos_outliers.groupby('DS_CARGO').count().reset_index()

#Ordena o DataFrame resultante em ordem decrescente com base na coluna 'VR_DESPESA_MAX_CAMPANHA'
agrupado_por_cargo = agrupado_por_cargo.sort_values(by='VR_DESPESA_MAX_CAMPANHA', ascending=False)

# Exibe o total de eleitos por cargo
print("Total de eleitos por CARGO - df_eleitos_dados_demograficos_outliers:")
print(agrupado_por_cargo)

#Agrupa os dados por 'TP_AGREMIACAO' e calcula a contagem de ocorrências (eleitos) para o segundo gráfico
agrupado_por_agremiacao = df_eleitos_dados_demograficos_outliers.groupby('TP_AGREMIACAO').count().reset_index()

#Ordena o DataFrame resultante em ordem decrescente com base na coluna 'VR_DESPESA_MAX_CAMPANHA'
agrupado_por_agremiacao = agrupado_por_agremiacao.sort_values(by='VR_DESPESA_MAX_CAMPANHA', ascending=False)

# Exibe o total de eleitos por agremiação
print("Total de eleitos por AGREMIACAO - df_eleitos_dados_demograficos_outliers:")
print(agrupado_por_agremiacao)

#Cria uma figura com dois subplots (gráficos lado a lado)
fig, axes = plt.subplots(1, 2, figsize=(8, 10))

#Plota o primeiro gráfico de barras (por cargo)
sns.barplot(data=agrupado_por_cargo, x='DS_CARGO', y='VR_DESPESA_MAX_CAMPANHA', ax=axes[0],
            order=agrupado_por_cargo['DS_CARGO'])
axes[0].set_title('Total de eleitos por CARGO - df_eleitos_dados_demograficos_outliers')
axes[0].set_xlabel('CARGO')
axes[0].set_ylabel('Total de eleitos')
axes[0].tick_params(axis='x', rotation=90)
axes[0].ticklabel_format(axis='y', style='plain', useOffset=False)

#Plota o segundo gráfico de barras (por agremiação)
sns.barplot(data=agrupado_por_agremiacao, x='TP_AGREMIACAO', y='VR_DESPESA_MAX_CAMPANHA', ax=axes[1],
            order=agrupado_por_agremiacao['TP_AGREMIACAO'])
axes[1].set_title('Total de eleitos por AGREMIACAO - df_eleitos_dados_demograficos_outliers')
axes[1].set_xlabel('AGREMIACAO')
axes[1].set_ylabel('Total de eleitos')
axes[1].tick_params(axis='x', rotation=90)
axes[1].ticklabel_format(axis='y', style='plain', useOffset=False)

#Ajusta o layout dos subplots para melhor visualização
plt.tight_layout()

#Exibe os gráficos
plt.show()

print('*' * 100)
print('FIM Análise Outliers')
print('*' * 100)

print('*' * 100)
print('FIM Análise e Exploração dos Dados - Criação de Modelos de Machine Learning')
print('*' * 100)


