"""
model_training.py: 
Responsável por:
  - Carregar e pré-processar o dataset de risco de crédito
  - Treinar o modelo XGBoost
  - Avaliar o modelo (AUC-ROC, classification report)
  - Salvar o modelo e o dataset processado.

Uso direto:
    python src/model_training.py --csv data/credit_risk_dataset.csv

Uso como módulo:
    from src.model_training import load_dataset, train_model
    dataset = load_dataset("data/credit_risk_dataset.csv")
    model, X_test, y_test = train_model(dataset)

"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("model_training")

# Constantes
CATEGORICAL_COLS = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

TARGET_COL  = "loan_status"
TEST_SIZE   = 0.20
RANDOM_SEED = 42

MODEL_PATH   = Path("model/xgboost.pkl")
DATASET_PATH = Path("model/dataset.pkl")


# Carregamento e pré-processamento 
def load_dataset(csv_path: str) -> dict:
    """
    Carrega o CSV, remove NaNs e codifica features categóricas.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Dataset carregado: %d linhas, %d colunas", len(df), df.shape[1])

    # Remove linhas com valores ausentes
    before = len(df)
    df = df.dropna()
    if len(df) < before:
        logger.info("Removidas %d linhas com NaN", before - len(df))

    df_original = df.copy()

    # Encoding de variáveis categóricas
    encoders = {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            logger.warning("Coluna '%s' não encontrada — pulando encoding", col)
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        logger.info("Encoded: %s → %s", col, list(le.classes_))

    # Split features / target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    feature_names = list(X.columns)

    # Split treino / teste estratificado
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )

    default_rate = y.mean() * 100
    logger.info(
        "Split: %d treino / %d teste | taxa de default: %.1f%%",
        len(X_train), len(X_test), default_rate
    )

    return {
        "X_train":      X_train,
        "X_test":       X_test,
        "y_train":      y_train,
        "y_test":       y_test,
        "feature_names": feature_names,
        "encoders":     encoders,
        "df_original":  df_original,
    }


# 2. Treinamento
def train_model(dataset: dict) -> tuple:
    """
    Treina o XGBoost com os dados do dataset.

    Retorna:
        model   : XGBClassifier treinado
        X_test  : features de teste (para avaliação externa)
        y_test  : labels de teste
    """
    X_train = dataset["X_train"]
    y_train = dataset["y_train"]
    X_test  = dataset["X_test"]
    y_test  = dataset["y_test"]

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
        verbosity=0,
    )

    logger.info("Treinando XGBoost...")
    model.fit(X_train, y_train)
    logger.info("Treinamento concluído.")

    return model, X_test, y_test


# 3. Avaliação 
def evaluate_model(model, X_test, y_test) -> dict:
    """
    Avalia o modelo no conjunto de teste.
    """
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    auc    = roc_auc_score(y_test, y_pred_prob)
    report = classification_report(
        y_test, y_pred,
        target_names=["Adimplente", "Default"]
    )

    logger.info("AUC-ROC: %.4f", auc)
    print("\nAUC-ROC:", round(auc, 4))
    print(report)

    return {"auc": auc, "report": report, "y_pred": y_pred}


# 4. Persistência 
def save_model(model, dataset: dict, model_path=MODEL_PATH, dataset_path=DATASET_PATH):
    """
    Salva o modelo e o dataset processado em disco.
    Usar load_model() para carregar.
    """
    model_path   = Path(model_path)
    dataset_path = Path(dataset_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Modelo salvo em: %s", model_path)

    # Salva apenas o necessário para as outras etapas (sem X_train/y_train)
    dataset_to_save = {k: v for k, v in dataset.items()
                       if k not in ("X_train", "y_train")}
    with open(dataset_path, "wb") as f:
        pickle.dump(dataset_to_save, f)
    logger.info("Dataset processado salvo em: %s", dataset_path)


def load_model(model_path=MODEL_PATH, dataset_path=DATASET_PATH) -> tuple:
    """
    Carrega o modelo e o dataset do disco.

    Retorna:
        model   : XGBClassifier treinado
        dataset : dict com X_test, y_test, feature_names, encoders, df_original
    """
    model_path   = Path(model_path)
    dataset_path = Path(dataset_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {model_path}\n"
            "Execute primeiro: python src/model_training.py --csv data/credit_risk_dataset.csv"
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    logger.info("Modelo carregado de: %s", model_path)
    return model, dataset


#  5. 
def main():
    parser = argparse.ArgumentParser(
        description="Treina e salva o modelo XGBoost."
    )
    parser.add_argument("--csv",          required=True,        help="Caminho para o CSV do dataset")
    parser.add_argument("--model-path",   default=MODEL_PATH,   help="Onde salvar o modelo (.pkl)")
    parser.add_argument("--dataset-path", default=DATASET_PATH, help="Onde salvar o dataset processado (.pkl)")
    parser.add_argument("--no-save",      action="store_true",  help="Não salva em disco (só avalia)")
    args = parser.parse_args()

    dataset = load_dataset(args.csv)
    model, X_test, y_test = train_model(dataset)
    evaluate_model(model, X_test, y_test)

    if not args.no_save:
        save_model(model, dataset, args.model_path, args.dataset_path)
        print(f"\nModelo salvo em:  {args.model_path}")
        print(f"Dataset salvo em: {args.dataset_path}")


if __name__ == "__main__":
    main()
