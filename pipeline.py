"""
pipeline.py — Pipeline CLI para o Projeto de Explicabilidade Adaptativa com LIME

Comandos disponíveis:
    train     — Treina o modelo XGBoost e salva em disco
    explain   — Gera uma explicação adaptativa para um cliente do conjunto de teste
    evaluate  — Avalia o LIME adaptativo em lote e salva CSVs e gráficos em results/

Exemplos de uso:
    python pipeline.py train --csv data/credit_risk_dataset.csv
    python pipeline.py explain --idx 0
    python pipeline.py explain --idx 5 --with-slm
    python pipeline.py evaluate --n 50 --output results/
    python pipeline.py evaluate --n 100 --output results/ --start-samples 50 --max-samples 2000
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.model_training import load_dataset, train_model, evaluate_model, save_model, load_model
from src.adaptive_lime import AdaptiveLime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("pipeline")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_adaptive_lime(dataset: dict) -> AdaptiveLime:
    """Constrói o AdaptiveLime usando os dados de treino do dataset."""
    feature_names = dataset["feature_names"]
    categorical_cols = list(dataset["encoders"].keys())
    categorical_idx = [
        feature_names.index(c) for c in categorical_cols if c in feature_names
    ]
    background_data = dataset["X_train"].values
    return AdaptiveLime(
        background_data=background_data,
        feature_names=feature_names,
        categorical_features=categorical_idx,
        class_names=["Adimplente", "Default"],
    )


def _predict_client(model, client_data: np.ndarray) -> tuple[float, str]:
    """Retorna (probabilidade_default, status_string)."""
    prob = model.predict_proba([client_data])[0][1]
    classe = model.predict([client_data])[0]
    status = "Default" if classe == 1 else "Adimplente"
    return prob, status


# ──────────────────────────────────────────────────────────────────────────────
# Comando: train
# ──────────────────────────────────────────────────────────────────────────────

def cmd_train(args):
    """Treina o XGBoost e salva modelo + dataset processado."""
    print(f"[train] Carregando dataset: {args.csv}")
    dataset = load_dataset(args.csv)

    print("[train] Treinando XGBoost...")
    model, X_test, y_test = train_model(dataset)

    print("[train] Avaliando modelo...")
    evaluate_model(model, X_test, y_test)

    model_path = Path(args.model_path)
    dataset_path = Path(args.dataset_path)
    save_model(model, dataset, model_path, dataset_path)
    print(f"\n[train] Modelo salvo em:  {model_path}")
    print(f"[train] Dataset salvo em: {dataset_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Comando: explain
# ──────────────────────────────────────────────────────────────────────────────

def cmd_explain(args):
    """Explica uma instância do conjunto de teste usando LIME adaptativo."""
    print("[explain] Carregando modelo e dataset...")
    try:
        model, dataset = load_model()
    except FileNotFoundError as exc:
        print(f"[explain] ERRO: {exc}")
        sys.exit(1)

    X_test = dataset["X_test"]
    if args.idx >= len(X_test):
        print(f"[explain] ERRO: índice {args.idx} fora do intervalo (0-{len(X_test) - 1}).")
        sys.exit(1)

    client_data = X_test.iloc[args.idx].values
    prob, status = _predict_client(model, client_data)

    print(f"\n[explain] Cliente {args.idx}")
    print(f"  Probabilidade de Default: {prob:.2%}")
    print(f"  Classe Predita:           {status}")

    adaptive_lime = _build_adaptive_lime(dataset)

    semantic_fn = None
    lime_to_text_fn = None
    if args.with_slm:
        try:
            from src.slm.semantic_arbiter import check_convergence, lime_to_text
            semantic_fn = check_convergence
            lime_to_text_fn = lime_to_text
            print("[explain] Árbitro semântico (Qwen2.5) habilitado.")
        except ImportError:
            print("[explain] AVISO: Módulo SLM não disponível. Continuando sem árbitro.")

    print("\n[explain] Iniciando LIME adaptativo...")
    exp, samples_used = adaptive_lime.explain_instance(
        data_row=client_data,
        predict_fn=model.predict_proba,
        start_samples=args.start_samples,
        max_samples=args.max_samples,
        step_multiplier=args.step_multiplier,
        r2_threshold=args.r2_threshold,
        coef_tol=args.coef_tol,
        semantic_check_fn=semantic_fn,
        lime_to_text_fn=lime_to_text_fn,
        prediction=status,
        prob=prob,
    )

    print(f"\n[explain] Resultado:")
    print(f"  Amostras utilizadas: {samples_used}")
    print(f"  R² final:            {exp.score:.4f}")
    print("\n  Top-5 Features:")
    for feature_desc, peso in exp.as_list()[:5]:
        sinal = "+" if peso > 0 else "-"
        print(f"    [{sinal}] {feature_desc}: {peso:.4f}")

    if args.with_slm:
        print("\n[explain] Gerando explicação em português via Qwen2.5 (aguarde)...")
        try:
            from src.slm.explanation_generator import generate_explanation
            texto = generate_explanation(
                prediction=status,
                probability=prob,
                lime_features=exp.as_list()[:5],
            )
            print("\n" + "=" * 60)
            print(texto)
            print("=" * 60)
        except ConnectionError as exc:
            print(f"\n[explain] ERRO DE CONEXÃO: {exc}")
        except TimeoutError as exc:
            print(f"\n[explain] TIMEOUT: {exc}")


# ──────────────────────────────────────────────────────────────────────────────
# Comando: evaluate
# ──────────────────────────────────────────────────────────────────────────────

def cmd_evaluate(args):
    """
    Avalia o LIME adaptativo em lote sobre N instâncias do conjunto de teste.

    Salva:
      - results/evaluation_results.csv   — métricas por instância
      - results/summary.json             — estatísticas agregadas
      - results/samples_distribution.png — histograma de perturbações
      - results/r2_distribution.png      — histograma de R²
      - results/convergence_rate.png     — taxa de convergência
      - results/top_features.png         — features mais frequentes no top-3
    """
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[evaluate] Carregando modelo e dataset...")
    try:
        model, dataset = load_model()
    except FileNotFoundError as exc:
        print(f"[evaluate] ERRO: {exc}")
        sys.exit(1)

    X_test = dataset["X_test"]
    y_test = dataset["y_test"]
    feature_names = dataset["feature_names"]

    n = min(args.n, len(X_test))
    if args.n > len(X_test):
        print(
            f"[evaluate] AVISO: apenas {len(X_test)} instâncias disponíveis; "
            f"avaliando todas."
        )

    adaptive_lime = _build_adaptive_lime(dataset)

    # ── Avaliação em lote ──────────────────────────────────────────────────
    records = []
    top3_counter: Counter = Counter()

    print(f"\n[evaluate] Avaliando {n} instâncias...")
    for i in range(n):
        client_data = X_test.iloc[i].values
        prob, status = _predict_client(model, client_data)
        true_label = int(y_test.iloc[i])

        t0 = time.time()
        exp, samples_used = adaptive_lime.explain_instance(
            data_row=client_data,
            predict_fn=model.predict_proba,
            start_samples=args.start_samples,
            max_samples=args.max_samples,
            step_multiplier=args.step_multiplier,
            r2_threshold=args.r2_threshold,
            coef_tol=args.coef_tol,
        )
        elapsed = time.time() - t0

        r2 = exp.score
        converged = samples_used < args.max_samples

        # Top-3 feature names (sem o intervalo de valor, apenas o nome base)
        top3_raw = exp.as_list()[:3]
        top3_names = [_extract_feature_name(feat, feature_names) for feat, _ in top3_raw]
        for name in top3_names:
            top3_counter[name] += 1

        record = {
            "instance_id":   i,
            "true_label":    true_label,
            "predicted":     status,
            "probability":   round(prob, 4),
            "samples_used":  samples_used,
            "r2_score":      round(r2, 4),
            "converged":     converged,
            "elapsed_sec":   round(elapsed, 2),
            "top1_feature":  top3_names[0] if len(top3_names) > 0 else "",
            "top2_feature":  top3_names[1] if len(top3_names) > 1 else "",
            "top3_feature":  top3_names[2] if len(top3_names) > 2 else "",
        }
        records.append(record)

        flag = "✓" if converged else "✗"
        print(
            f"  [{flag}] idx={i:4d} | amostras={samples_used:5d} | "
            f"R²={r2:.4f} | {status} ({prob:.2%}) | {elapsed:.1f}s"
        )

    # ── Salvar CSV ─────────────────────────────────────────────────────────
    csv_path = output_dir / "evaluation_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"\n[evaluate] CSV salvo: {csv_path}")

    # ── Estatísticas agregadas ─────────────────────────────────────────────
    df = pd.DataFrame(records)
    convergence_rate = df["converged"].mean()
    summary = {
        "n_instances":       n,
        "convergence_rate":  round(float(convergence_rate), 4),
        "samples_mean":      round(float(df["samples_used"].mean()), 2),
        "samples_median":    float(df["samples_used"].median()),
        "samples_min":       int(df["samples_used"].min()),
        "samples_max":       int(df["samples_used"].max()),
        "r2_mean":           round(float(df["r2_score"].mean()), 4),
        "r2_median":         round(float(df["r2_score"].median()), 4),
        "r2_min":            round(float(df["r2_score"].min()), 4),
        "r2_max":            round(float(df["r2_score"].max()), 4),
        "elapsed_total_sec": round(float(df["elapsed_sec"].sum()), 2),
        "default_rate":      round(float((df["predicted"] == "Default").mean()), 4),
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[evaluate] Resumo salvo: {summary_path}")

    # ── Gráficos ───────────────────────────────────────────────────────────
    _plot_samples_distribution(df, output_dir)
    _plot_r2_distribution(df, output_dir)
    _plot_convergence_rate(df, output_dir)
    _plot_top_features(top3_counter, output_dir)

    # ── Exibir resumo no terminal ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  RESUMO DA AVALIAÇÃO")
    print("=" * 55)
    print(f"  Instâncias avaliadas : {summary['n_instances']}")
    print(f"  Taxa de convergência : {summary['convergence_rate']:.1%}")
    print(f"  Perturbações  média  : {summary['samples_mean']:.1f}")
    print(f"  Perturbações  mediana: {summary['samples_median']:.0f}")
    print(f"  R²  médio            : {summary['r2_mean']:.4f}")
    print(f"  R²  mediano          : {summary['r2_median']:.4f}")
    print(f"  Tempo total          : {summary['elapsed_total_sec']:.1f}s")
    print("=" * 55)
    print(f"\n  Arquivos gerados em: {output_dir.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# Funções auxiliares de extração e plotagem
# ──────────────────────────────────────────────────────────────────────────────

def _extract_feature_name(feature_str: str, feature_names: list) -> str:
    """
    Extrai o nome base de uma string LIME do tipo 'loan_amnt <= 5000.00'
    encontrando qual feature_name está contido na string.
    """
    for name in sorted(feature_names, key=len, reverse=True):
        if name in feature_str:
            return name
    return feature_str.split(" ")[0]


def _plot_samples_distribution(df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["samples_used"], bins=20, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Número de perturbações utilizadas")
    ax.set_ylabel("Número de instâncias")
    ax.set_title("Distribuição de Perturbações — LIME Adaptativo")
    ax.axvline(df["samples_used"].mean(), color="#DD4444", linestyle="--",
               label=f"Média: {df['samples_used'].mean():.0f}")
    ax.axvline(df["samples_used"].median(), color="#44AA44", linestyle=":",
               label=f"Mediana: {df['samples_used'].median():.0f}")
    ax.legend()
    plt.tight_layout()
    path = output_dir / "samples_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Gráfico salvo: {path}")


def _plot_r2_distribution(df: pd.DataFrame, output_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["r2_score"], bins=20, color="#55A868", edgecolor="white", alpha=0.85)
    ax.set_xlabel("R² do LIME")
    ax.set_ylabel("Número de instâncias")
    ax.set_title("Distribuição do R² — LIME Adaptativo")
    ax.axvline(df["r2_score"].mean(), color="#DD4444", linestyle="--",
               label=f"Média: {df['r2_score'].mean():.4f}")
    ax.axvline(df["r2_score"].median(), color="#4C72B0", linestyle=":",
               label=f"Mediana: {df['r2_score'].median():.4f}")
    ax.legend()
    plt.tight_layout()
    path = output_dir / "r2_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Gráfico salvo: {path}")


def _plot_convergence_rate(df: pd.DataFrame, output_dir: Path):
    converged = df["converged"].sum()
    not_converged = len(df) - converged
    labels = ["Convergiu", "Não convergiu"]
    values = [converged, not_converged]
    colors = ["#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(val),
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_ylabel("Número de instâncias")
    ax.set_title(
        f"Taxa de Convergência: {converged / len(df):.1%} "
        f"({converged}/{len(df)})"
    )
    plt.tight_layout()
    path = output_dir / "convergence_rate.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Gráfico salvo: {path}")


def _plot_top_features(counter: Counter, output_dir: Path, top_k: int = 10):
    if not counter:
        return
    most_common = counter.most_common(top_k)
    features, counts = zip(*most_common)

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("muted", len(features))
    bars = ax.barh(list(features)[::-1], list(counts)[::-1],
                   color=palette[::-1], edgecolor="white")
    ax.set_xlabel("Frequência no Top-3")
    ax.set_title("Features Mais Frequentes no Top-3 — LIME Adaptativo")
    for bar, val in zip(bars, list(counts)[::-1]):
        ax.text(
            val + 0.1,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
        )
    plt.tight_layout()
    path = output_dir / "top_features.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Gráfico salvo: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Parser principal
# ──────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Pipeline CLI — Explicabilidade Adaptativa com LIME",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────────────
    p_train = sub.add_parser("train", help="Treina o modelo XGBoost")
    p_train.add_argument(
        "--csv", required=True,
        help="Caminho para o CSV do dataset (ex: data/credit_risk_dataset.csv)",
    )
    p_train.add_argument(
        "--model-path", default="model/xgboost.pkl",
        help="Destino para salvar o modelo (.pkl)",
    )
    p_train.add_argument(
        "--dataset-path", default="model/dataset.pkl",
        help="Destino para salvar o dataset processado (.pkl)",
    )

    # ── explain ────────────────────────────────────────────────────────────
    p_explain = sub.add_parser(
        "explain",
        help="Explica uma instância do conjunto de teste",
    )
    p_explain.add_argument(
        "--idx", type=int, default=0,
        help="Índice da instância no conjunto de teste (padrão: 0)",
    )
    p_explain.add_argument(
        "--with-slm", action="store_true",
        help="Usa Qwen2.5 (Docker Model Runner) para gerar texto em português",
    )
    _add_lime_args(p_explain)

    # ── evaluate ───────────────────────────────────────────────────────────
    p_eval = sub.add_parser(
        "evaluate",
        help="Avalia o LIME adaptativo em lote e gera relatórios em results/",
    )
    p_eval.add_argument(
        "--n", type=int, default=50,
        help="Número de instâncias a avaliar (padrão: 50)",
    )
    p_eval.add_argument(
        "--output", default="results/",
        help="Diretório de saída para CSVs e gráficos (padrão: results/)",
    )
    _add_lime_args(p_eval)

    return parser


def _add_lime_args(subparser: argparse.ArgumentParser):
    """Adiciona argumentos de configuração do LIME a um sub-parser."""
    subparser.add_argument(
        "--start-samples", type=int, default=50,
        help="Perturbações iniciais do LIME (padrão: 50)",
    )
    subparser.add_argument(
        "--max-samples", type=int, default=5000,
        help="Limite máximo de perturbações (padrão: 5000)",
    )
    subparser.add_argument(
        "--step-multiplier", type=float, default=2.0,
        help="Multiplicador entre iterações (padrão: 2.0)",
    )
    subparser.add_argument(
        "--r2-threshold", type=float, default=0.70,
        help="Limiar de R² para convergência (padrão: 0.70)",
    )
    subparser.add_argument(
        "--coef-tol", type=float, default=0.05,
        help="Tolerância de variação de coeficientes (padrão: 0.05)",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Ponto de entrada
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "explain":
        cmd_explain(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
