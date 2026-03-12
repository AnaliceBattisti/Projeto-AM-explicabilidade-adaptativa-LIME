# Pipeline CLI — Guia de Uso

Este documento descreve como utilizar o `pipeline.py` para treinar o modelo, gerar explicações individuais e realizar avaliações em lote.

---

## Pré-requisitos

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. (Opcional) Para explicações em texto via SLM, configure o Docker Model Runner:
   - Abra o Docker Desktop
   - Habilite: `Settings > AI > Enable Docker Model Runner`
   - Habilite: `Settings > AI > Enable host-side TCP support`
   - Baixe o modelo: `docker model pull ai/qwen2.5:7B-Q4_0`

---

## Comandos

### `train` — Treinar o modelo

Treina o XGBoost com o dataset de risco de crédito e salva o modelo em `model/`.

```bash
python pipeline.py train --csv data/credit_risk_dataset.csv
```

**Opções:**
| Argumento | Padrão | Descrição |
|---|---|---|
| `--csv` | *(obrigatório)* | Caminho para o arquivo CSV do dataset |
| `--model-path` | `model/xgboost.pkl` | Destino para salvar o modelo |
| `--dataset-path` | `model/dataset.pkl` | Destino para salvar o dataset processado |

---

### `explain` — Explicar uma instância

Gera uma explicação LIME adaptativa para uma instância específica do conjunto de teste.

```bash
# Instância padrão (índice 0)
python pipeline.py explain

# Instância específica
python pipeline.py explain --idx 5

# Com explicação em português via Qwen2.5 (requer Docker Model Runner)
python pipeline.py explain --idx 5 --with-slm
```

**Opções:**
| Argumento | Padrão | Descrição |
|---|---|---|
| `--idx` | `0` | Índice da instância no conjunto de teste |
| `--with-slm` | `False` | Gera texto explicativo em português via Qwen2.5 |
| `--start-samples` | `50` | Perturbações iniciais do LIME |
| `--max-samples` | `5000` | Limite máximo de perturbações |
| `--step-multiplier` | `2.0` | Multiplicador entre iterações |
| `--r2-threshold` | `0.70` | Limiar de R² para convergência |
| `--coef-tol` | `0.05` | Tolerância de variação de coeficientes |

---

### `evaluate` — Avaliação em lote

Avalia o LIME adaptativo em `N` instâncias do conjunto de teste e gera relatórios.

```bash
# Avaliar 50 instâncias (padrão)
python pipeline.py evaluate

# Avaliar 100 instâncias
python pipeline.py evaluate --n 100

# Avaliar com parâmetros customizados
python pipeline.py evaluate --n 200 --start-samples 100 --max-samples 2000
```

**Opções:**
| Argumento | Padrão | Descrição |
|---|---|---|
| `--n` | `50` | Número de instâncias a avaliar |
| `--output` | `results/` | Diretório de saída para arquivos gerados |
| `--start-samples` | `50` | Perturbações iniciais do LIME |
| `--max-samples` | `5000` | Limite máximo de perturbações |
| `--step-multiplier` | `2.0` | Multiplicador entre iterações |
| `--r2-threshold` | `0.70` | Limiar de R² para convergência |
| `--coef-tol` | `0.05` | Tolerância de variação de coeficientes |

**Arquivos gerados em `results/`:**
| Arquivo | Descrição |
|---|---|
| `evaluation_results.csv` | Métricas por instância: amostras usadas, R², convergência, features |
| `summary.json` | Estatísticas agregadas: média, mediana, taxa de convergência |
| `samples_distribution.png` | Histograma de perturbações utilizadas |
| `r2_distribution.png` | Histograma dos valores de R² |
| `convergence_rate.png` | Gráfico de barras: instâncias que convergiram vs. não convergiram |
| `top_features.png` | Features mais frequentes no Top-3 das explicações |

---

## Fluxo completo de exemplo

```bash
# 1. Treinar o modelo
python pipeline.py train --csv data/credit_risk_dataset.csv

# 2. Explicar um cliente específico
python pipeline.py explain --idx 10 --with-slm

# 3. Avaliar em lote e gerar relatórios
python pipeline.py evaluate --n 100 --output results/
```

---

## Estrutura dos arquivos de saída

### `evaluation_results.csv`

| Coluna | Descrição |
|---|---|
| `instance_id` | Índice da instância no conjunto de teste |
| `true_label` | Rótulo verdadeiro (0=Adimplente, 1=Default) |
| `predicted` | Classe predita pelo XGBoost |
| `probability` | Probabilidade de default |
| `samples_used` | Perturbações usadas até convergência |
| `r2_score` | R² do modelo LIME na convergência |
| `converged` | Booleano: atingiu critério de convergência? |
| `elapsed_sec` | Tempo de execução em segundos |
| `top1_feature` | Feature mais importante |
| `top2_feature` | Segunda feature mais importante |
| `top3_feature` | Terceira feature mais importante |

### `summary.json`

```json
{
  "n_instances": 50,
  "convergence_rate": 0.72,
  "samples_mean": 183.4,
  "samples_median": 100.0,
  "samples_min": 50,
  "samples_max": 5000,
  "r2_mean": 0.3821,
  "r2_median": 0.3654,
  "r2_min": 0.1203,
  "r2_max": 0.4279,
  "elapsed_total_sec": 142.5,
  "default_rate": 0.22
}
```
