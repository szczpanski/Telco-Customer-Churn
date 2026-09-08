# Telco Customer Churn

Projeto para identificar clientes com maior risco de cancelamento (*churn*) em uma empresa de telecomunicações.

A solução cobre análise exploratória, preparação dos dados, comparação de modelos, validação cruzada, tuning de hiperparâmetros, avaliação em holdout, análise de ranking, interpretabilidade e disponibilização dos resultados em Streamlit.

## Contexto de negócio

O objetivo é estimar a probabilidade de churn e priorizar clientes para possíveis ações de retenção.

O modelo responde **quais clientes apresentam maior risco de churn**, mas não mede diretamente qual ação de retenção causará permanência. Uma evolução natural é combinar o score com experimentação controlada ou *uplift modeling*.

## Objetivos

- entender e preparar os dados;
- realizar EDA de churners vs. não churners;
- comparar Logistic Regression, Random Forest e XGBoost;
- avaliar modelos com e sem balanceamento;
- realizar tuning;
- selecionar o champion por validação cruzada;
- avaliar o modelo final em holdout;
- calcular ROC-AUC, PR-AUC, Precision, Recall e F1;
- calcular Lift@K e Capture Rate@K;
- analisar thresholds;
- interpretar o modelo com SHAP;
- disponibilizar resultados em Streamlit.

## Dataset

A base contém:

- **7.043 clientes**
- **21 colunas**
- **19 features utilizadas na modelagem**
- target: `Churn`
- churn: **35,16%**

### Qualidade

- **0 registros duplicados**
- **0 valores ausentes**
- `customerID` único por linha
- `customerID` excluído da modelagem

## Principais resultados da EDA

### Contract

| Contrato | Taxa de churn |
|---|---:|
| Month-to-month | **43,50%** |
| One year | **23,37%** |
| Two year | **22,35%** |

`Contract` foi a variável com maior associação com churn.

### PaymentMethod

| Método | Taxa de churn |
|---|---:|
| Electronic check | **42,59%** |
| Bank transfer (automatic) | 33,16% |
| Credit card (automatic) | 32,43% |
| Mailed check | 32,27% |

### Cramér's V

| Feature | Cramér's V |
|---|---:|
| Contract | **0,212** |
| PaymentMethod | **0,089** |
| DeviceProtection | 0,010 |
| PaperlessBilling | 0,009 |
| OnlineSecurity | 0,008 |

### Variáveis numéricas

| Feature | Não churn | Churn |
|---|---:|---:|
| tenure | 36,41 | 35,60 |
| MonthlyCharges | 70,33 | 69,46 |
| TotalCharges | 2.562,14 | 2.474,50 |

As diferenças entre churners e não churners foram pequenas nas variáveis numéricas quando analisadas isoladamente.

## Pré-processamento

O pré-processamento foi encapsulado em `Pipeline` para evitar *data leakage*.

### Numéricas
- imputação pela mediana;
- `StandardScaler`.

### Categóricas
- imputação pela moda;
- `OneHotEncoder(handle_unknown="ignore")`.

`SeniorCitizen` foi tratado como variável categórica/binária.

## Estratégia de validação

- holdout estratificado de **20%**;
- treino: **5.634 clientes**;
- teste: **1.409 clientes**;
- churn treino: **35,16%**;
- churn teste: **35,13%**;
- `StratifiedKFold` com **5 folds**.

## Modelos avaliados

- Logistic Regression;
- Logistic Regression balanceada;
- Random Forest;
- Random Forest balanceada;
- XGBoost;
- XGBoost balanceado.

Também foi realizado tuning de Random Forest e XGBoost.

## Comparação dos modelos

| Modelo | ROC-AUC CV |
|---|---:|
| **Logistic Regression** | **0,6289** |
| Logistic Regression Balanced | 0,6289 |
| XGBoost Balanced | 0,6095 |
| XGBoost | 0,6092 |
| Random Forest | 0,5978 |
| Random Forest Balanced | 0,5978 |

Após tuning:

| Modelo | ROC-AUC CV |
|---|---:|
| **Logistic Regression** | **0,6289** |
| Random Forest Tuned | 0,6266 |
| XGBoost Tuned | 0,6257 |

O aumento de complexidade não trouxe ganho consistente.

## Champion

**Logistic Regression**

### Holdout

| Métrica | Resultado |
|---|---:|
| ROC-AUC | **0,6275** |
| PR-AUC | **0,4463** |
| Accuracy | 0,6473 |
| Precision | 0,4941 |
| Recall | 0,1697 |
| F1-score | 0,2526 |

O ROC-AUC do holdout ficou muito próximo do cross-validation, indicando estabilidade da avaliação.

## Threshold

A análise exploratória mostrou que o threshold padrão de `0.50` reduz bastante o recall.

| Threshold | Precision | Recall | F1 | % base acionada |
|---|---:|---:|---:|---:|
| 0,20 | 0,3760 | 0,9556 | 0,5396 | 89,28% |
| 0,25 | 0,4126 | **0,8343** | **0,5521** | 71,04% |
| 0,30 | 0,4270 | 0,7737 | 0,5503 | 63,66% |
| 0,40 | 0,4344 | 0,5354 | 0,4796 | 43,29% |
| 0,50 | 0,4941 | 0,1697 | 0,2526 | 12,07% |

O threshold de produção deve refletir capacidade operacional e função de custo, não apenas o maior F1.

> **Nota metodológica:** nesta versão, a tabela de thresholds foi construída sobre o holdout para análise exploratória. Em uma versão final, a escolha do threshold deve ser feita via validação cruzada ou conjunto de validação separado, preservando o holdout exclusivamente para avaliação final.

## Lift e Capture Rate

| Top da base | Lift | Capture Rate |
|---|---:|---:|
| 5% | 1,403x | 7,07% |
| 10% | **1,393x** | **13,94%** |
| 20% | **1,403x** | **28,08%** |
| 30% | 1,272x | 38,18% |
| 40% | 1,206x | 48,28% |

Exemplo: abordando os 20% de clientes com maior score, o modelo concentra cerca de **28% dos churners**, com taxa de churn aproximadamente **1,40x** acima da média.

## Feature Engineering

Foram avaliadas:

- `avg_charge_per_month`;
- `num_active_services`;
- faixas de `tenure`.

ROC-AUC CV com feature engineering: **0,6261**

Como não houve ganho sobre o baseline de **0,6289**, essas features não foram mantidas.

## Interpretabilidade

O modelo final foi interpretado com **SHAP**.

A interpretação é preditiva, não causal. Os achados da EDA indicam que `Contract` e `PaymentMethod` merecem atenção especial.

## Insights de negócio

1. Clientes `Month-to-month` representam o principal segmento de risco.
2. `Electronic check` apresenta churn superior aos demais métodos.
3. Modelos mais complexos não superaram a Regressão Logística.
4. O poder preditivo parece limitado pelas features disponíveis.
5. Para campanhas com capacidade limitada, o ranking é mais útil que um threshold fixo.
6. Melhorias futuras provavelmente dependem mais de novas fontes comportamentais do que de novos algoritmos.

## Limitações

- dataset relativamente pequeno;
- features principalmente cadastrais e contratuais;
- desempenho discriminatório moderado;
- avaliação offline;
- ausência de variáveis comportamentais recentes;
- ausência de função de custo operacional;
- risco de churn não mede efeito causal da retenção.

## Próximos passos

- incorporar dados de uso e comportamento;
- adicionar histórico de atendimento/reclamações;
- adicionar histórico de pagamento;
- avaliar calibração;
- definir threshold por função de custo;
- monitorar drift;
- registrar experimentos com MLflow;
- validar campanhas com A/B Testing;
- avaliar uplift modeling.

## Estrutura do projeto

```text
telco-churn-challenge/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── EDA.ipynb
│   └── Modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── metrics.py
│   ├── train_model.py
│   └── evaluate_model.py
├── models/
│   └── best_model.joblib
├── reports/
│   ├── figures/
│   ├── Decision_Log.md
│   └── metrics.json
├── tests/
│   └── test_preprocessing.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── app.py
├── Dockerfile
├── Makefile
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```

## Como executar

### Ambiente

```bash
conda create -n telco-churn python=3.11 -y
conda activate telco-churn
pip install -r requirements.txt
```

No macOS, se o XGBoost reclamar de `libomp`:

```bash
conda install -c conda-forge llvm-openmp -y
```

### Dataset

Coloque:

```text
telco_customer_churn.xlsx
```

em:

```text
data/raw/telco_customer_churn.xlsx
```

### Testes

```bash
pytest -q
```

### Treinamento

```bash
make train
```

ou:

```bash
python src/train_model.py --input data/raw/telco_customer_churn.xlsx
```

### Avaliação

```bash
python src/evaluate_model.py
```

### Dashboard

```bash
streamlit run app.py
```

## Tecnologias

- Python 3.11
- pandas
- NumPy
- scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Streamlit
- pytest
- Jupyter
- Conda
- GitHub Actions
- Docker

## Conclusão

A Regressão Logística apresentou o melhor resultado entre os modelos avaliados, com ROC-AUC próximo de **0,63** e desempenho estável entre validação cruzada e holdout.

Random Forest e XGBoost não trouxeram ganho consistente, mesmo após tuning.

Neste dataset, o principal limitador parece ser a quantidade de informação preditiva disponível nas features, e não a complexidade do algoritmo.

O score pode ser usado para **priorização de risco**, principalmente quando combinado com métricas de ranking como Lift@K. Para transformar o score em impacto real de retenção, recomenda-se validar ações com experimentação controlada e, futuramente, uplift modeling.
