# Previsão de Preços de Casas (California Housing)

Projeto de introdução a Machine Learning: prever o preço mediano das casas de um bairro da Califórnia a partir das suas características, comparando três modelos de regressão com scikit-learn.

## Objetivo

Construir e avaliar modelos que, dadas as características de um bairro (rendimento dos habitantes, idade das casas, localização, entre outras), estimem o preço mediano das casas. O projeto percorre o ciclo completo de um problema de aprendizagem supervisionada: análise exploratória, preparação dos dados, treino, avaliação e interpretação dos resultados.

## Dataset

[California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html), disponível no scikit-learn e baseado no censo norte-americano de 1990.

- 20 640 bairros, 8 variáveis de entrada e 1 variável-alvo
- Não tem valores em falta
- **Alvo:** `MedHouseVal`, o preço mediano das casas, em centenas de milhares de dólares

| Variável | Descrição |
|---|---|
| `MedInc` | Rendimento mediano do bairro |
| `HouseAge` | Idade mediana das casas |
| `AveRooms` | Média de divisões por casa |
| `AveBedrms` | Média de quartos por casa |
| `Population` | População do bairro |
| `AveOccup` | Média de ocupantes por casa |
| `Latitude` / `Longitude` | Localização do bairro |

## Metodologia

1. **Análise exploratória:** distribuições, matriz de correlação e mapa de preços por localização.
2. **Divisão dos dados:** 80% para treino e 20% para teste (`random_state=42`).
3. **Regressão linear** como modelo de referência.
4. **Limpeza de dados:** remoção, no treino, dos preços cortados e dos valores extremos das variáveis `AveRooms`, `AveBedrms`, `AveOccup` e `Population` (acima do percentil 99).
5. **Árvore de decisão**, com estudo do efeito da profundidade no sobreajuste.
6. **Random Forest** com 100 árvores e análise da importância das variáveis.

Métricas usadas: erro absoluto médio (**MAE**) e coeficiente de determinação (**R²**), calculadas sempre no conjunto de teste.

## Resultados

| Modelo | MAE | R² |
|---|---|---|
| Regressão linear | 0.533 | 0.576 |
| Árvore de decisão (sem limite) | 0.455 | 0.622 |
| **Random Forest** | **0.328** | **0.805** |

O preço está em centenas de milhares de dólares, pelo que o MAE do Random Forest corresponde a um erro médio de cerca de 33 000 $.

### Efeito da limpeza de dados na regressão linear

| Teste avaliado em | MAE antes | MAE depois | R² antes | R² depois |
|---|---|---|---|---|
| Todos os bairros | 0.533 | 0.513 | 0.576 | 0.376 |
| Sem preço cortado | 0.488 | 0.470 | 0.544 | 0.239 |
| Bairros "normais" | 0.481 | 0.442 | 0.587 | 0.640 |

A limpeza melhorou o modelo nos bairros típicos, mas deixou-o frágil perante valores fora do intervalo de treino: o modelo limpo chegou a prever preços negativos (mínimo de -24) em bairros com valores extremos, o que fez o R² global cair. Mostra que a regressão linear é sensível a valores extremos e que uma melhoria numa métrica ou numa parte dos dados pode esconder uma degradação noutra.

### Sobreajuste numa árvore de decisão

![R² por profundidade da árvore](images/profundidade_arvore.png)

Até uma profundidade de 8 a 10, a árvore melhora no treino e no teste. A partir daí, o R² de treino continua a subir até perto de 1.0, enquanto o de teste desce: a árvore deixa de aprender padrões e passa a decorar os dados.

### Importância das variáveis (Random Forest)

![Importância das variáveis](images/importancia_variaveis.png)

<!-- Confirmar a ordem com o gráfico real antes de publicar -->
O rendimento mediano (`MedInc`) é a variável mais relevante, seguida da ocupação média por casa e da localização (`Latitude` e `Longitude`), o que concorda com o mapa de preços, onde as zonas costeiras concentram os valores mais altos.

## Principais conclusões

- O Random Forest reduziu o erro em cerca de 38% face à regressão linear e explica cerca de 80% da variação dos preços no teste.
- Combinar muitas árvores diminui a instabilidade de uma árvore isolada, embora ainda exista algum sobreajuste.
- A limpeza de dados exige medir o seu efeito: neste caso melhorou os casos típicos e piorou os extremos.

## Limitações

- Os preços do dataset estão cortados em 500 000 $, o que limita a previsão das casas mais caras.
- Os dados são de 1990 e não representam o mercado atual.
- A profundidade da árvore foi analisada com base no conjunto de teste, o que o torna ligeiramente menos independente. A abordagem mais rigorosa seria usar validação cruzada.
- Os hiperparâmetros do Random Forest não foram afinados.

## Como reproduzir

**No Google Colab:** abrir o notebook em `notebooks/` e executar todas as células.

**Localmente:**

```bash
git clone <endereço-do-repositório>
cd <pasta-do-repositório>
pip install -r requirements.txt
jupyter notebook notebooks/previsao_precos_casas.ipynb
```

## Estrutura do repositório

```
├── images/          # gráficos usados neste README
├── notebooks/       # notebook com a análise e os modelos
├── requirements.txt
├── LICENSE
└── README.md
```

## Possíveis melhorias

- Afinar os hiperparâmetros com validação cruzada.
- Testar modelos de boosting, como Gradient Boosting ou XGBoost.
- Criar novas variáveis, por exemplo a distância ao litoral.
- Tratar o corte dos preços com métodos específicos para dados censurados.

## Tecnologias

Python, pandas, matplotlib, seaborn, scikit-learn, Jupyter / Google Colab.

## Autor

Rafael Baptista
