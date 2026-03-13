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
        max_samples=10000,
        step_multiplier=2,
        r2_threshold=0.75,  # mantido apenas como referência no relatório; não controla parada
        coef_tol=0.05,
        stable_required=2,
        semantic_check_fn=None,
        lime_to_text_fn=None,
        prediction=None,
        prob=None,
    ):
        """
        Gera a explicação de forma adaptativa, aumentando o número de perturbações
        até que os critérios de convergência sejam atingidos.

        A convergência é determinada por ESTABILIDADE:
          - coef_diff <= coef_tol  (coeficientes estáveis entre iterações)
          - Top-3 features idênticas às da iteração anterior
          - Ambas as condições válidas por `stable_required` iterações consecutivas

        O R² é reportado como métrica de qualidade da aproximação local,
        mas não bloqueia nem ativa a convergência.

        Retorna: (exp, current_samples, reason, arbiter_responses)
        """
        current_samples = start_samples

        prev_coefs = None
        prev_top3 = None
        texto_anterior = None
        arbiter_responses = []
        consecutive_stable = 0

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
                top3_match = (set(current_top3) == set(prev_top3))

                print(
                    f"   R²: {current_r2:.4f} | Diff Coefs: {coef_diff:.4f} "
                    f"| Top-3 Match: {top3_match} | Estável: {consecutive_stable}/{stable_required}"
                )

                if coef_diff <= coef_tol and top3_match:
                    consecutive_stable += 1

                    if consecutive_stable >= stable_required:
                        if semantic_check_fn is not None and texto_anterior is not None:
                            texto_atual = lime_to_text_fn(exp.as_list()[:5], prediction, prob)
                            resultado = semantic_check_fn(texto_anterior, texto_atual)
                            print(f"   Árbitro: {resultado['reason']} (confiança: {resultado['confidence']:.2f})")

                            arbiter_responses.append({
                                "samples": current_samples,
                                "reason": resultado.get("reason", ""),
                                "raw_response": resultado.get("raw_response", ""),
                                "confidence": resultado.get("confidence", 0.0)
                            })

                            if resultado["converged"] and resultado["confidence"] >= 0.7:
                                print(f"Convergência atingida com {current_samples} perturbações! (R²={current_r2:.4f})")
                                return exp, current_samples, "semantic_converged", arbiter_responses
                            else:
                                print(f"   Árbitro discordou — continuando...")
                                consecutive_stable = 0
                        else:
                            print(f"Convergência atingida com {current_samples} perturbações! (R²={current_r2:.4f})")
                            return exp, current_samples, "stability_converged", arbiter_responses
                else:
                    consecutive_stable = 0

            prev_coefs = current_coefs
            prev_top3 = current_top3

            if lime_to_text_fn is not None and prediction is not None and prob is not None:
                texto_anterior = lime_to_text_fn(exp.as_list()[:5], prediction, prob)

            current_samples *= step_multiplier

        print("Aviso: Número máximo de perturbações atingido sem convergência total.")
        return exp, current_samples // step_multiplier, "max_samples_reached", arbiter_responses

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
