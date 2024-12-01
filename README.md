# Implementação e Análise do Algoritmo de K-means com o Dataset Human Activity Recognition

## Descrição

Este projeto implementa o algoritmo de **K-means** para agrupar atividades humanas com base em dados coletados de sensores de smartphones. O dataset utilizado é o "Human Activity Recognition Using Smartphones" disponível no repositório UCI. O objetivo é realizar a análise exploratória dos dados, aplicar o algoritmo de K-means e avaliar a qualidade dos clusters formados.

## Estrutura do Projeto

O projeto é dividido nas seguintes pastas:

- **data/**: Contém o dataset original.
- **scripts/**: Contém o código Python utilizado para a análise e aplicação do algoritmo.
  
## Como Executar

1. Clone o repositório:
    ```bash
    git clone <https://github.com/jvsantos1/Projeto-K-Means>
    ```

2. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

3. Coloque o dataset na pasta `data/UCI HAR Dataset/`.

4. Execute o script:
    ```bash
    python scripts/index.py
    ```

## Resultados

Os principais resultados incluem:

1. **Gráfico do Método do Cotovelo**: Para escolher o número ideal de clusters.
2. **Distribuição dos Dados**: Visualização dos dados após a redução de dimensionalidade (PCA).
3. **Visualização dos Clusters**: Mostra os clusters formados e seus centroides.

## Referências

- [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones)
- [Artigo sobre o dataset](https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf)

## Autores

Projeto desenvolvido por: 
**Arthur Lago Martis**.
**João Victor Oliveira Santos**.

