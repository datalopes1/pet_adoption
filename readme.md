# Predição da Adoção de Pets 🐶

O Pet Adoption Dataset fornece uma visão compreensiva sobre vários fatores que influenciam a probabilidade de um pet ser adotado de um abrigo. O dataset inclui informações detalhadas sobre animais disponíveis para adoção, cobrindo várias caraterísticas e atributos.

![cachorro](https://github.com/datalopes1/pet_adoption/blob/main/doc/img/jana-shnipelson-j_Z5N9p_XOc-unsplash.jpg?raw=true)

### Objetivos e resultados
O objetivo deste projeto é fazer uma breve análise exploratória para entender os fatores que levam ao aumento da probabilidade de adoção dos pets, e criar um modelo com foco de prever a probabilidade de adoção dos pets. 

O algoritmo escolhido foi o Logistic Regression, e o modelo teve como resultados uma Acurácia de 90,96% e AUC Score de 90,51%. 

### 🛠️ Ferramentas utilizadas
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)

## Estrutura do Dataset
|Coluna|Descrição|
|-------|--------|
|PetID|Identificador único de cada pet|
|PetType|Tipo do pet (cachorro, gato, etc.)|
|Breed|Raça especifica de cada pet|
|AgeMonths|Idade do pet em meses|
|Color|Cor do pet|
|Size|Tamanho do animal (pequeno, médio, grande)|
|WeightKg|Peso em kg|
|Vaccinated|Status de vacinação|
|HealthCondition|Se o pet é saudável|
|TimeInShelterDays|Quanto tempo em dias que o pet está no abrigo|
|AdoptionFee|Taxa de adoção|
|PreviousOwner|Se o pet teve um dono antes ou não|
|AdoptionLikelihood|Probabilidade do pet ser adotado|

## Bibliotecas Python utilizadas
#### Manipulação de dados
- Pandas, Numpy.
#### EDA
- Seaborn, Matplotlib.
#### Machine Learning e Feature Engineering
- sklearn, feature_engine

# Análise Exploratória de Dados 
### Tipo de Animal
![](https://github.com/datalopes1/pet_adoption/blob/main/doc/img/plots/plot1.png?raw=true)

### Raça do Animal
![](https://github.com/datalopes1/pet_adoption/blob/main/doc/img/plots/plot2.png?raw=true)

### Idade em meses
![](https://github.com/datalopes1/pet_adoption/blob/main/doc/img/plots/plot3.png?raw=true)

### Cor
![](https://github.com/datalopes1/pet_adoption/blob/main/doc/img/plots/plot4.png?raw=true)

### Tamanho
![](https://github.com/datalopes1/pet_adoption/blob/main/doc/img/plots/plot5.png?raw=true)

![](https://github.com/datalopes1/pet_adoption/blob/main/doc/img/plots/plot6.png?raw=true)

### Peso
![](https://github.com/datalopes1/pet_adoption/blob/main/doc/img/plots/plot7.png?raw=true)

### Vacinação
![](https://github.com/datalopes1/pet_adoption/blob/main/doc/img/plots/plot8.png?raw=true)

### Condições de Saúde
![](https://github.com/datalopes1/pet_adoption/blob/main/doc/img/plots/plot9.png?raw=true)

### Tempo de abrigo
![](https://github.com/datalopes1/pet_adoption/blob/main/doc/img/plots/plot10.png?raw=true)

### Taxa de Adoção
![](https://github.com/datalopes1/pet_adoption/blob/main/doc/img/plots/plot11.png?raw=true)

### Já Teve Outros Donos?
![](https://github.com/datalopes1/pet_adoption/blob/main/doc/img/plots/plot12.png?raw=true)

## Conclusões pós-EDA

- Cães são os animais de estimação mais adotados; 
- Periquitos estão na segunda posição;
- Fatores de saúde do animal tem forte influência na decisão de adotar ou não;
- O preço da taxa de adoção também faz diferença na adoção. 

![gato](https://github.com/datalopes1/pet_adoption/blob/main/doc/img/alisa-matthews-ZtHbp39rjIM-unsplash.jpg?raw=true)

# Modelo de Predição

### Métricas dos Modelos Avaliados
#### Acurácia 

|Modelo|Pontuação|
|------|---------|
|Random Forest Treino|0.9454|
|Random Forest Teste|0.9314|
|Descision Tree Treino|0.9563|
|Decision Tree Teste|0.9158|
|Logistic Regression Treino|0.9166|
|Logistic Regression Teste|0.9096|

O modelo com maior acurácia em testes foi o Random Forest, mas o que apresenta o melhor ajuste é o Logistic Regression. O modelo Decision Tree tem uma diferença próxima à 0,05 sendo o pior ajuste. 
#### ROC AUC

|Modelo|Pontuação|
|------|---------|
|Random Forest Treino|0.9552|
|Random Forest Teste|0.9080|
|Descision Tree Treino|0.9867|
|Decision Tree Teste|0.9057|
|Logistic Regression Treino|0.9414|
|Logistic Regression Teste|0.9051|

O modelo com o maior Score AUC foi a Decision Tree mas ela apresenta uma grande diferença entre treino e teste (acima dos 0,05) e não tem o melhor ajuste. Novamente neste critério, o melhor ajuste foi do modelo Logistic Regression. Que portando será o modelo escolhido.

![periquito](https://github.com/datalopes1/pet_adoption/blob/main/doc/img/dim-hou-ZAtcN0f9HJc-unsplash.jpg?raw=true)