import numpy as np
import pandas as pd
from src.model_training import load_model
from src.adaptive_lime import AdaptiveLime
from src.slm.explanation_generator import generate_explanation
from src.slm.semantic_arbiter import lime_to_text, check_convergence

def main():
    print("1. Carregando modelo e dataset...")
    try:
        model, dataset = load_model()
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        return
    
    feature_names = dataset["feature_names"]
    
    categorical_cols_names = list(dataset["encoders"].keys())
    categorical_features_idx = [
        feature_names.index(col) for col in categorical_cols_names if col in feature_names
    ]

    background_data = dataset["X_train"].values

    print("\n2. Inicializando o AdaptiveLime...")
    adaptive_lime = AdaptiveLime(
        background_data=background_data,
        feature_names=feature_names,
        categorical_features=categorical_features_idx,
        class_names=["Adimplente", "Default"]
    )

    cliente_idx = 0
    cliente_data = dataset["X_test"].iloc[cliente_idx].values
    
    prob_default = model.predict_proba([cliente_data])[0][1]
    classe_predita = model.predict([cliente_data])[0]
    status = "Default" if classe_predita == 1 else "Adimplente"
    
    print(f"\nExplicando Cliente {cliente_idx}")
    print(f"Probabilidade de Default (XGBoost): {prob_default:.2%}")
    print(f"Classe Predita: {status}")
    
    exp, amostras_usadas, razao_convergencia = adaptive_lime.explain_instance(
        data_row=cliente_data,
        predict_fn=model.predict_proba,
        start_samples=50,
        max_samples=5000,
        step_multiplier=2,
        r2_threshold=0.10,  
        coef_tol=0.05,
        semantic_check_fn=check_convergence,
        lime_to_text_fn=lime_to_text,
        prediction=status,
        prob=prob_default,
    )

    print(f"\nResultado Final:")
    print(f"Utilizadas {amostras_usadas} amostras. Motivo: {razao_convergencia}")
    print(f"R² Final: {exp.score:.4f}")
    
    features = exp.as_list()[:5]
    
    print("\nTop Features que influenciaram a decisão:")
    for feature_desc, peso in exp.as_list()[:5]:
        sinal = "+" if peso > 0 else "-"
        print(f"  [{sinal}] {feature_desc}: {peso:.4f}")
        
    print("\nGerando explicação em português (aguarde até 60s)...")
    try:
        texto = generate_explanation(
            prediction=status,
            probability=prob_default,
            lime_features=features,
        )

        print("\nExplicação para o cliente:")
        print("=" * 60)
        print(texto)
        print("=" * 60)
    except ConnectionError as e:
        print(f"\nERRO DE CONEXÃO: {e}")
    except TimeoutError as e:
        print(f"\nTIMEOUT: {e}")

if __name__ == "__main__":
    main()