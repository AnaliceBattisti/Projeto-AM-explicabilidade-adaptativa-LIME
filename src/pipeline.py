import argparse
import sys
import time
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Adjust path to import modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.model_training import load_model, train_model, load_dataset, evaluate_model, save_model
from src.adaptive_lime import AdaptiveLime
from src.slm.semantic_arbiter import check_convergence, lime_to_text
from src.slm.explanation_generator import generate_explanation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] pipeline — %(message)s"
)
logger = logging.getLogger("pipeline")

RESULTS_DIR = Path("results")
DOCS_DIR = Path("docs")

# Configuração global do matplotlib para gráficos publicáveis
matplotlib.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
})


def ensure_dirs():
    RESULTS_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)


# ─────────────────────────── TRAIN ───────────────────────────

def train_command(args):
    logger.info("Iniciando treinamento...")
    dataset = load_dataset(args.csv)
    model, X_test, y_test = train_model(dataset)
    evaluate_model(model, X_test, y_test)

    if not args.no_save:
        save_model(model, dataset)


# ─────────────────────────── EXPLAIN ───────────────────────────

def explain_command(args):
    logger.info(f"Iniciando explicação em lote para {args.num_instances} instâncias...")

    try:
        model, dataset = load_model()
    except (FileNotFoundError, TypeError, ImportError, AttributeError) as e:
        logger.warning(f"Modelo não encontrado ou incompatível ({e}). Retreinando automaticamente...")
        args_train = argparse.Namespace(csv="data/credit_risk_dataset.csv", no_save=False)
        train_command(args_train)
        model, dataset = load_model()

    feature_names = dataset["feature_names"]
    categorical_cols_names = list(dataset["encoders"].keys())

    categorical_features_idx = [
        feature_names.index(col) for col in categorical_cols_names
        if col in feature_names
    ]

    background_data = dataset["X_train"].values

    adaptive_lime = AdaptiveLime(
        background_data=background_data,
        feature_names=feature_names,
        categorical_features=categorical_features_idx,
        class_names=["Adimplente", "Default"]
    )

    X_test = dataset["X_test"]

    if args.random:
        instances_to_explain = X_test.sample(
            n=min(args.num_instances, len(X_test)), random_state=42
        )
    else:
        instances_to_explain = X_test.head(args.num_instances)

    results = []
    start_time_all = time.time()

    for idx, (original_idx, row) in enumerate(instances_to_explain.iterrows()):
        data_row = row.values
        logger.info(
            f"--- Processando instância {idx + 1}/{len(instances_to_explain)} "
            f"(ID original: {original_idx}) ---"
        )

        start_time_instance = time.time()

        # Predição
        prob_default = model.predict_proba([data_row])[0][1]
        classe_predita = model.predict([data_row])[0]
        status = "Default" if classe_predita == 1 else "Adimplente"

        # LIME Adaptativo
        try:
            exp, samples_used, reason = adaptive_lime.explain_instance(
                data_row=data_row,
                predict_fn=model.predict_proba,
                start_samples=args.start_samples,
                max_samples=args.max_samples,
                step_multiplier=2,
                r2_threshold=args.r2_threshold,
                coef_tol=0.05,
                semantic_check_fn=check_convergence if not args.no_slm else None,
                lime_to_text_fn=lime_to_text if not args.no_slm else None,
                prediction=status,
                prob=prob_default
            )

            final_r2 = exp.score

            # Top-5 features para registro
            top_features = "; ".join(
                f"{feat}: {w:.4f}" for feat, w in exp.as_list()[:5]
            )

        except Exception as e:
            logger.error(f"Erro ao explicar instância {original_idx}: {e}")
            samples_used = -1
            final_r2 = -1.0
            reason = f"error: {str(e)}"
            top_features = ""

        elapsed_time = time.time() - start_time_instance

        results.append({
            "original_idx": original_idx,
            "predicted_class": status,
            "probability_default": round(prob_default, 6),
            "samples_used": samples_used,
            "final_r2": round(final_r2, 6),
            "convergence_reason": reason,
            "execution_time_sec": round(elapsed_time, 4),
            "used_slm": not args.no_slm,
            "top_features": top_features,
        })

    total_time = time.time() - start_time_all
    logger.info(f"Processamento concluído em {total_time:.2f}s")

    # Salvar resultados
    df_results = pd.DataFrame(results)
    output_file = RESULTS_DIR / f"explanation_results_{int(time.time())}.csv"
    df_results.to_csv(output_file, index=False)
    logger.info(f"Resultados salvos em {output_file}")

    # Gerar gráficos se solicitado
    if args.plot:
        plot_results(output_file)

    return output_file


# ─────────────────────────── EVALUATE ───────────────────────────

def evaluate_command(args):
    logger.info("Gerando gráficos e avaliações...")
    if args.input_csv:
        csv_path = Path(args.input_csv)
    else:
        csvs = sorted(
            RESULTS_DIR.glob("explanation_results_*.csv"),
            key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not csvs:
            logger.error("Nenhum arquivo de resultados encontrado em results/")
            return
        csv_path = csvs[0]
        logger.info(f"Usando arquivo mais recente: {csv_path}")

    plot_results(csv_path)


# ─────────────────────────── GRÁFICOS E RELATÓRIO ───────────────────────────

def plot_results(csv_path):
    """Gera gráficos publicáveis e relatório a partir do CSV de resultados."""
    df = pd.read_csv(csv_path)
    timestamp = int(time.time())

    # Filtrar linhas com erro
    df_valid = df[df["samples_used"] > 0].copy()

    if df_valid.empty:
        logger.warning("Nenhuma instância válida para plotar.")
        return

    sns.set_style("whitegrid")
    palette = sns.color_palette("Set2")

    # ── 1. Histograma de Perturbações ──
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df_valid["samples_used"], bins=15, kde=True, color=palette[0], ax=ax)
    median_val = df_valid["samples_used"].median()
    ax.axvline(median_val, color="red", linestyle="--", linewidth=1.5,
               label=f"Mediana: {median_val:.0f}")
    ax.set_title("Distribuição do Número de Perturbações (LIME Adaptativo)")
    ax.set_xlabel("Número de Perturbações")
    ax.set_ylabel("Frequência")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS_DIR / f"hist_samples_{timestamp}.png")
    plt.close(fig)

    # ── 2. Scatter R² vs Perturbações (colorido por convergência) ──
    fig, ax = plt.subplots(figsize=(9, 5))
    reason_labels = {
        "r2_converged": "Convergência R²",
        "semantic_converged": "Convergência Semântica",
        "max_samples_reached": "Limite Atingido",
    }
    df_valid["convergence_label"] = df_valid["convergence_reason"].map(
        lambda r: reason_labels.get(r, r)
    )
    sns.scatterplot(
        data=df_valid, x="samples_used", y="final_r2",
        hue="convergence_label", style="convergence_label",
        s=80, ax=ax
    )
    r2_threshold = 0.10
    ax.axhline(r2_threshold, color="gray", linestyle="--", alpha=0.7,
               label=f"Limiar R² = {r2_threshold}")
    ax.set_title("Estabilidade (R²) vs Custo Computacional")
    ax.set_xlabel("Número de Perturbações")
    ax.set_ylabel("R² Final do LIME")
    ax.legend(title="Convergência")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / f"scatter_r2_samples_{timestamp}.png")
    plt.close(fig)

    # ── 3. Boxplot: Perturbações por Tipo de Convergência ──
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(
        data=df_valid, x="convergence_label", y="samples_used",
        hue="convergence_label", palette="Set2", legend=False, ax=ax
    )
    ax.set_title("Perturbações Necessárias por Tipo de Convergência")
    ax.set_xlabel("Tipo de Convergência")
    ax.set_ylabel("Número de Perturbações")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / f"boxplot_samples_convergence_{timestamp}.png")
    plt.close(fig)

    # ── 4. Histograma de Tempo de Execução ──
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(df_valid["execution_time_sec"], bins=15, kde=True,
                 color=palette[2], ax=ax)
    mean_time = df_valid["execution_time_sec"].mean()
    ax.axvline(mean_time, color="red", linestyle="--", linewidth=1.5,
               label=f"Média: {mean_time:.2f}s")
    ax.set_title("Distribuição do Tempo de Execução por Instância")
    ax.set_xlabel("Tempo (segundos)")
    ax.set_ylabel("Frequência")
    ax.legend()
    fig.tight_layout()
    fig.savefig(DOCS_DIR / f"hist_execution_time_{timestamp}.png")
    plt.close(fig)

    # ── 5. Barplot: Contagem por Motivo de Convergência ──
    fig, ax = plt.subplots(figsize=(8, 5))
    reason_counts = df_valid["convergence_label"].value_counts()
    bars = ax.bar(reason_counts.index, reason_counts.values, color=palette[:len(reason_counts)])
    for bar, val in zip(bars, reason_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", fontweight="bold")
    ax.set_title("Motivos de Convergência")
    ax.set_xlabel("Tipo")
    ax.set_ylabel("Quantidade de Instâncias")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / f"bar_convergence_reasons_{timestamp}.png")
    plt.close(fig)

    # ── 6. Scatter: Tempo vs Perturbações ──
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.scatterplot(
        data=df_valid, x="samples_used", y="execution_time_sec",
        hue="convergence_label", style="convergence_label",
        s=80, ax=ax
    )
    ax.set_title("Tempo de Execução vs Perturbações")
    ax.set_xlabel("Número de Perturbações")
    ax.set_ylabel("Tempo de Execução (s)")
    ax.legend(title="Convergência")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / f"scatter_time_samples_{timestamp}.png")
    plt.close(fig)

    # ── 7. Painel Resumo (2x2) ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 7a - Histograma perturbações
    sns.histplot(df_valid["samples_used"], bins=15, kde=True,
                 color=palette[0], ax=axes[0, 0])
    axes[0, 0].set_title("Distribuição de Perturbações")
    axes[0, 0].set_xlabel("Perturbações")

    # 7b - R² vs Perturbações
    sns.scatterplot(
        data=df_valid, x="samples_used", y="final_r2",
        hue="convergence_label", s=60, ax=axes[0, 1], legend=False
    )
    axes[0, 1].axhline(r2_threshold, color="gray", linestyle="--", alpha=0.7)
    axes[0, 1].set_title("R² vs Perturbações")

    # 7c - Tempo de execução
    sns.histplot(df_valid["execution_time_sec"], bins=15, kde=True,
                 color=palette[2], ax=axes[1, 0])
    axes[1, 0].set_title("Distribuição de Tempo")
    axes[1, 0].set_xlabel("Tempo (s)")

    # 7d - Barras de convergência
    rc = df_valid["convergence_label"].value_counts()
    axes[1, 1].bar(rc.index, rc.values, color=palette[:len(rc)])
    axes[1, 1].set_title("Motivos de Convergência")
    for tick in axes[1, 1].get_xticklabels():
        tick.set_rotation(15)

    fig.suptitle("Resumo — LIME Adaptativo", fontweight="bold")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / f"panel_summary_{timestamp}.png")
    plt.close(fig)

    # ── Relatório de Texto (UTF-8) ──
    report_path = DOCS_DIR / f"summary_report_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Relatório de Execução — LIME Adaptativo\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"Arquivo fonte: {csv_path}\n")
        f.write(f"Total de instâncias: {len(df)}\n")
        f.write(f"Instâncias válidas: {len(df_valid)}\n\n")

        f.write(f"--- Métricas de Perturbações ---\n")
        f.write(f"Média:   {df_valid['samples_used'].mean():.1f}\n")
        f.write(f"Mediana: {df_valid['samples_used'].median():.1f}\n")
        f.write(f"Mínimo:  {df_valid['samples_used'].min()}\n")
        f.write(f"Máximo:  {df_valid['samples_used'].max()}\n\n")

        f.write(f"--- Métricas de R² ---\n")
        f.write(f"Média:   {df_valid['final_r2'].mean():.4f}\n")
        f.write(f"Mediana: {df_valid['final_r2'].median():.4f}\n")
        f.write(f"Mínimo:  {df_valid['final_r2'].min():.4f}\n")
        f.write(f"Máximo:  {df_valid['final_r2'].max():.4f}\n\n")

        f.write(f"--- Tempo de Execução ---\n")
        f.write(f"Tempo médio por instância: {df_valid['execution_time_sec'].mean():.2f}s\n")
        f.write(f"Tempo total estimado: {df_valid['execution_time_sec'].sum():.2f}s\n\n")

        f.write(f"--- Motivos de Convergência ---\n")
        for reason, count in reason_counts.items():
            pct = count / len(df_valid) * 100
            f.write(f"  {reason}: {count} ({pct:.1f}%)\n")

    logger.info(f"Gráficos (7 figuras) e relatório salvos em {DOCS_DIR}/")
    logger.info(f"Relatório: {report_path}")


# ─────────────────────────── CLI ───────────────────────────

def main():
    ensure_dirs()

    parser = argparse.ArgumentParser(
        description="Pipeline de Explicabilidade Adaptativa — LIME + XGBoost + Qwen2.5"
    )
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponíveis")

    # Subparser: train
    parser_train = subparsers.add_parser("train", help="Treina o modelo XGBoost")
    parser_train.add_argument("--csv", default="data/credit_risk_dataset.csv",
                              help="Caminho do dataset CSV")
    parser_train.add_argument("--no-save", action="store_true",
                              help="Não salvar modelo em disco")

    # Subparser: explain
    parser_explain = subparsers.add_parser("explain",
                                           help="Gera explicações em lote via LIME adaptativo")
    parser_explain.add_argument("--num_instances", type=int, default=50,
                                help="Número de clientes a explicar (padrão: 50)")
    parser_explain.add_argument("--start_samples", type=int, default=50,
                                help="Perturbações iniciais (padrão: 50)")
    parser_explain.add_argument("--max_samples", type=int, default=5000,
                                help="Limite máximo de perturbações (padrão: 5000)")
    parser_explain.add_argument("--r2_threshold", type=float, default=0.10,
                                help="Limiar de R² para convergência (padrão: 0.10)")
    parser_explain.add_argument("--random", action="store_true",
                                help="Amostragem aleatória do conjunto de teste")
    parser_explain.add_argument("--no_slm", action="store_true",
                                help="Desativar árbitro semântico (roda só critério matemático)")
    parser_explain.add_argument("--plot", action="store_true",
                                help="Gerar gráficos ao final da execução")

    # Subparser: evaluate
    parser_eval = subparsers.add_parser("evaluate",
                                         help="Gera gráficos e relatório a partir de CSV existente")
    parser_eval.add_argument("--input_csv", required=False,
                             help="Caminho do CSV de resultados (padrão: mais recente)")

    args = parser.parse_args()

    if args.command == "train":
        train_command(args)
    elif args.command == "explain":
        explain_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
