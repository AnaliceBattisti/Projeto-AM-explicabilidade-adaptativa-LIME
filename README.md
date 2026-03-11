# Adaptive LIME — Explicabilidade Adaptativa para Risco de Crédito

Projeto de aprendizado de máquina com foco em **explicabilidade local adaptativa**. O objetivo é explicar as decisões de um modelo de crédito de forma coerente e eficiente, escolhendo automaticamente o número mínimo de perturbações necessárias para que a explicação do **LIME** seja estável.

O pipeline combina três componentes: um modelo XGBoost treinado sobre dados reais de crédito, o algoritmo LIME com convergência adaptativa, e o Qwen2.5 nano como árbitro semântico das explicações geradas.

---

## Estrutura do projeto

```
├── src/
│   ├── model_training.py     # pré-processamento, treino e avaliação do XGBoost
├── data/
│   └── credit_risk_dataset.csv   # dataset
├── model/                        # gerado após o treino (não versionado)
│   ├── xgboost.pkl
│   └── dataset.pkl
├── docs/                         # documentação e relatório
└── requirements.txt
```

---

## Dataset

O projeto usa o [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset), com 32.581 registros de solicitantes de crédito e taxa de inadimplência de 21,9%.

As features utilizadas são: idade, renda anual, tempo de emprego, valor do empréstimo, taxa de juros, comprometimento de renda, histórico de crédito, tipo de moradia, finalidade do empréstimo, grau de risco e histórico de inadimplência.

Download: https://www.kaggle.com/datasets/laotse/credit-risk-dataset

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Modelo base (XGBoost)

O modelo black box é um classificador XGBoost treinado com divisão 80/20. As variáveis categóricas são codificadas via `LabelEncoder` e linhas com valores ausentes são removidas.

Para treinar e salvar o modelo:

```bash
python src/model_training.py --csv data/credit_risk_dataset.csv
```

Isso gera dois arquivos em `model/`:
- `xgboost.pkl` — o modelo treinado
- `dataset.pkl` — o dataset processado com encoders e conjunto de teste

Os demais módulos do projeto carregam esses arquivos com:

```python
from src.model_training import load_model
model, dataset = load_model()
```
