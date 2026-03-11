import numpy as np
from lime.lime_tabular import LimeTabularExplainer

class AdaptiveLime:
    def __init__(self, background_data, feature_names, categorical_features, class_names=["Adimplente", "Default"]):
        """
        Inicializa o explainer base do LIME.
        """
        self.explainer = LimeTabularExplainer(
            training_data=background_data,
            feature_names=feature_names,
            categorical_features=categorical_features,
            class_names=class_names,
            mode='classification',
            kernel_width=3.0,
            random_state=42
        )

    def explain_instance(
        self, 
        data_row, 
        predict_fn, 
        start_samples=50, 
        max_samples=5000, 
        step_multiplier=2,
        r2_threshold=0.75, 
        coef_tol=0.05
    ):
        """
        Gera a explicação de forma adaptativa, aumentando o número de perturbações
        até que os critérios de convergência sejam atingidos.
        """
        current_samples = start_samples
        
        prev_coefs = None
        prev_top3 = None
        
        print(f"Iniciando explicação adaptativa para a instância...")

        while current_samples <= max_samples:
            print(f"Testando com {current_samples} perturbações...")
            
            exp = self.explainer.explain_instance(
                data_row=data_row,
                predict_fn=predict_fn,
                num_samples=current_samples
            )
            
            current_r2 = exp.score 

            current_exp_list = exp.local_exp[1]
            current_coefs = {idx: weight for idx, weight in current_exp_list}

            current_top3 = [
                idx for idx, weight in sorted(
                    current_exp_list, key=lambda x: abs(x[1]), reverse=True
                )[:3]
            ]
            
            if prev_coefs is not None:
                coef_diff = self._calculate_coef_diff(prev_coefs, current_coefs)
                
                top3_match = (current_top3 == prev_top3)
                
                print(f"   R²: {current_r2:.4f} | Diff Coefs: {coef_diff:.4f} | Top-3 Match: {top3_match}")
                
                if current_r2 >= r2_threshold and coef_diff <= coef_tol and top3_match:
                    print(f"Convergência atingida com {current_samples} perturbações!")
                    return exp, current_samples
            
            prev_coefs = current_coefs
            prev_top3 = current_top3
            
            current_samples *= step_multiplier
            
        print("Aviso: Número máximo de perturbações atingido sem convergência total.")
        return exp, current_samples // step_multiplier

    def _calculate_coef_diff(self, prev_coefs, current_coefs):
        """
        Calcula a diferença média absoluta entre os coeficientes de duas iterações.
        """
    
        all_keys = set(prev_coefs.keys()).union(set(current_coefs.keys()))
        
        diffs = []
        for k in all_keys:
            val_prev = prev_coefs.get(k, 0.0)
            val_curr = current_coefs.get(k, 0.0)
            diffs.append(abs(val_curr - val_prev))
            
        return np.mean(diffs)