# Adaptive LIME — Explicabilidade Adaptativa para Risco de Crédito

Projeto de aprendizado de máquina com foco em **explicabilidade local adaptativa**. O objetivo é explicar as decisões de um modelo de crédito de forma coerente e eficiente, escolhendo automaticamente o número mínimo de perturbações necessárias para que a explicação do **LIME** seja estável.

O pipeline combina três componentes: um modelo XGBoost treinado sobre dados reais de crédito, o algoritmo LIME com convergência adaptativa, e o Qwen2.5 como árbitro semântico das explicações geradas.

---

## Estrutura do projeto

```
├── src/
│   ├── model_training.py     # pré-processamento, treino e avaliação do XGBoost
│   ├── adaptive_lime.py      # algoritmo de convergência adaptativa
│   ├── pipeline.py           # orquestrador CLI (train, explain, evaluate)
│   └── slm/                  # árbitro semântico (Qwen2.5) e gerador de texto
│       ├── semantic_arbiter.py
│       └── explanation_generator.py
├── data/
│   └── credit_risk_dataset.csv   # dataset Kaggle Credit Risk
├── model/                        # gerado após o treino
│   ├── xgboost.pkl
│   └── dataset.pkl
├── results/                      # CSVs com métricas de cada execução em lote
├── docs/                         # gráficos gerados e relatórios
├── test_adaptive_lime.py         # teste de instância única (demo)
└── requirements.txt
```

---

## Dataset

O projeto usa o [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset), com 32.581 registros de solicitantes de crédito e taxa de inadimplência de ~21,9%.

**Features utilizadas:** idade, renda anual, tempo de emprego, valor do empréstimo, taxa de juros, comprometimento de renda, histórico de crédito, tipo de moradia, finalidade do empréstimo, grau de risco e histórico de inadimplência.

**Download:** https://www.kaggle.com/datasets/laotse/credit-risk-dataset

Coloque o arquivo CSV em `data/credit_risk_dataset.csv`.

---

## Pré-requisitos

- Python 3.10+
- (Opcional) Docker Desktop com Model Runner habilitado para o árbitro semântico (Qwen2.5)

---

## Instalação

```bash
# 1. Clonar o repositório
git clone <url-do-repo>
cd Projeto-AM-explicabilidade-adaptativa-LIME

# 2. Criar ambiente virtual (recomendado)
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows

# 3. Instalar dependências
pip install -r requirements.txt
```

### Configuração do Árbitro Semântico (opcional)

Para usar a convergência semântica com Qwen2.5:

1. Abra o Docker Desktop.
2. Ative em **Settings > AI > Enable Docker Model Runner**.
3. Ative **Settings > AI > Enable host-side TCP support**.
4. Baixe o modelo:
   ```bash
   docker model pull ai/qwen2.5:7B-Q4_0
   ```

---

## Como executar

O script principal é `src/pipeline.py`, que orquestra todo o fluxo via CLI.

### 1. Treinamento do modelo

Treina o XGBoost e salva os artefatos em `model/`.

```bash
python src/pipeline.py train --csv data/credit_risk_dataset.csv
```

| Argumento   | Descrição                  | Padrão                         |
| ----------- | -------------------------- | ------------------------------ |
| `--csv`     | Caminho do dataset CSV     | `data/credit_risk_dataset.csv` |
| `--no-save` | Não salvar modelo em disco | `False`                        |

### 2. Explicação em lote (LIME Adaptativo)

Gera explicações para múltiplos clientes, ajustando automaticamente o número de perturbações até convergir. Salva os resultados em `results/`.

```bash
# 50 clientes aleatórios com árbitro semântico (Qwen2.5) — modo padrão
python src/pipeline.py explain --num_instances 50 --random --plot

# Sem árbitro semântico (só critério matemático R²)
python src/pipeline.py explain --num_instances 50 --random --no_slm --plot
```

| Argumento         | Descrição                                    | Padrão  |
| ----------------- | -------------------------------------------- | ------- |
| `--num_instances` | Número de clientes a explicar                | `50`    |
| `--start_samples` | Perturbações iniciais                        | `50`    |
| `--max_samples`   | Limite máximo de perturbações                | `5000`  |
| `--r2_threshold`  | Limiar de R² para convergência               | `0.10`  |
| `--random`        | Amostragem aleatória do teste                | `False` |
| `--no_slm`        | Desativar árbitro semântico (só R²)          | `False` |
| `--plot`          | Gerar gráficos ao final                      | `False` |

Os resultados são salvos em `results/explanation_results_TIMESTAMP.csv` com as colunas:

| Coluna                | Descrição                               |
| --------------------- | --------------------------------------- |
| `original_idx`        | Índice original da instância no dataset |
| `predicted_class`     | Classe predita (Adimplente/Default)     |
| `probability_default` | Probabilidade de default                |
| `samples_used`        | Perturbações utilizadas                 |
| `final_r2`            | R² final do modelo linear local (LIME)  |
| `convergence_reason`  | Motivo de parada (r2/semantic/max)      |
| `execution_time_sec`  | Tempo de execução (segundos)            |
| `used_slm`            | Se usou o árbitro semântico             |
| `top_features`        | Top-5 features e seus pesos             |

### 3. Avaliação e gráficos

Se já existem CSVs gerados e você quer recriar os gráficos:

```bash
# Usar o CSV mais recente automaticamente
python src/pipeline.py evaluate

# Ou especificar um CSV
python src/pipeline.py evaluate --input_csv results/explanation_results_1773351748.csv
```

### Gráficos gerados

O comando `evaluate` (ou `explain --plot`) gera 7 visualizações em `docs/`:

| Arquivo                             | Descrição                                        |
| ----------------------------------- | ------------------------------------------------ |
| `hist_samples_*.png`                | Distribuição do número de perturbações           |
| `scatter_r2_samples_*.png`          | R² vs perturbações (colorido por convergência)   |
| `boxplot_samples_convergence_*.png` | Boxplot de perturbações por tipo de convergência |
| `hist_execution_time_*.png`         | Distribuição do tempo de execução                |
| `bar_convergence_reasons_*.png`     | Contagem de motivos de convergência              |
| `scatter_time_samples_*.png`        | Tempo de execução vs perturbações                |
| `panel_summary_*.png`               | Painel resumo 2x2 (ideal para artigo/slides)     |

Além disso, gera um `summary_report_*.txt` com estatísticas descritivas.

---

## Teste de instância única

Para testar a explicação de um único cliente (demo interativa):

```bash
python test_adaptive_lime.py
```

Este script carrega o modelo, explica o primeiro cliente do conjunto de teste e gera uma explicação em português via Qwen2.5.

---

## Arquivos gerados

| Diretório  | Conteúdo                                      |
| ---------- | --------------------------------------------- |
| `model/`   | Modelo XGBoost e dataset serializado (.pkl)   |
| `results/` | CSVs com métricas detalhadas de cada execução |
| `docs/`    | Gráficos (.png) e relatórios (.txt)           |

---

## Arquitetura do sistema

```
[Dataset CSV] ──► [model_training.py] ──► [XGBoost .pkl]
                                                │
                                                ▼
[Instância] ──► [adaptive_lime.py] ──► [Explicação LIME]
                     │                        │
               (loop adaptativo)              │
                     │                        ▼
          [semantic_arbiter.py] ◄──── [lime_to_text]
          (Qwen2.5 via Docker)           │
                     │                   ▼
                     └──────► [explanation_generator.py]
                              (Qwen2.5 via Docker)
                                    │
                                    ▼
                           [Texto em português]
```

**Convergência em 3 níveis:**

1. **Pré-filtro matemático:** R² ≥ limiar + estabilidade de coeficientes + top-3 features iguais (evita chamar a SLM desnecessariamente)
2. **Semântica (critério principal):** Árbitro SLM (Qwen2.5) confirma se a explicação em texto estabilizou entre as duas últimas iterações
3. **Fallback:** Limite máximo de perturbações atingido sem convergência
