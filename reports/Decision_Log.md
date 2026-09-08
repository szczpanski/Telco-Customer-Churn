# Decision Log

## Identificador
`customerID` é removido da modelagem por ser apenas um identificador da entidade.

## TotalCharges
O loader converte `TotalCharges` para numérico de forma segura. Valores inválidos são convertidos em `NaN` e imputados dentro do pipeline.

## Prevenção de leakage
O pré-processamento fica dentro de um `Pipeline`, de modo que imputação, escala e one-hot encoding são ajustados apenas nos folds de treino.

## Split e validação
Foi adotado holdout estratificado de 20% e 5-fold Stratified Cross-Validation no treino.

## Modelos
Foram comparados Logistic Regression, Random Forest e XGBoost, incluindo versões com balanceamento.

## Métrica de seleção
ROC-AUC médio em cross-validation é a métrica principal de seleção do champion. Também são acompanhados PR-AUC, Precision, Recall e F1.

## Threshold
O ponto de corte operacional deve ser definido no conjunto de treino, preferencialmente com previsões out-of-fold, preservando o holdout para avaliação final.

## Métricas de negócio
Lift@K e Capture Rate@K são usados para traduzir o ranking em capacidade de priorização operacional.

## Interpretabilidade
SHAP é usado para interpretação do modelo final. Importância preditiva não implica causalidade.
