import json
import re
import requests


SLM_URL   = "http://localhost:12434/engines/llama.cpp/v1/chat/completions"
SLM_MODEL = "ai/qwen2.5:7B-Q4_0"
TIMEOUT   = 120


def lime_to_text(lime_features_raw: list, prediction: str, prob: float) -> str:
    lines = []
    for item in lime_features_raw:
        if isinstance(item, tuple):
            feature_str, weight = item
            sign = "[+]" if weight > 0 else "[-]"
            line = f"{sign} {feature_str}: {weight:.4f}"
        elif isinstance(item, str):
            line = item
        else:
            raise ValueError(f"Item inválido em lime_features_raw: {item}")
        lines.append(line)

    features_text = "\n".join(f"  {line}" for line in lines)

    return (
        f"Predição: {prediction} (probabilidade de default: {prob:.1%})\n"
        f"Features mais importantes:\n"
        f"{features_text}"
    )


def check_convergence(explanation_a: str, explanation_b: str) -> dict:
    prompt = f"""Compare as duas explicações LIME abaixo, geradas para o mesmo cliente
com números diferentes de perturbações.

EXPLICAÇÃO A (menos perturbações):
{explanation_a}

EXPLICAÇÃO B (mais perturbações):
{explanation_b}

Avalie se convergiram considerando:
1. As top-3 features são as mesmas nas duas explicações?
2. Os sinais [+] e [-] de cada feature são iguais nas duas?
3. A conclusão geral sobre o cliente é equivalente?

Responda SOMENTE com este JSON, sem nenhum texto fora dele:
{{"converged": true ou false, "reason": "motivo em uma frase", "top_features_match": true ou false, "direction_match": true ou false, "confidence": 0.0 a 1.0}}"""

    try:
        response = requests.post(
            SLM_URL,
            json={
                "model": SLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.3,
            },
            timeout=TIMEOUT,
        )
        response.raise_for_status()

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Não foi possível conectar ao Docker Model Runner na porta 12434.\n"
            "Verifique:\n"
            "  1. Docker Desktop está aberto\n"
            "  2. Settings > AI > Enable Docker Model Runner está marcado\n"
            "  3. Settings > AI > Enable host-side TCP support está marcado\n"
            "  4. O modelo foi baixado: docker model pull ai/qwen2.5:7B-Q4_0"
        )
    except requests.exceptions.Timeout:
        raise TimeoutError(
            f"O Qwen2.5 não respondeu em {TIMEOUT}s. Tente novamente."
        )

    raw = response.json()["choices"][0]["message"]["content"]
    result = _parse_json(raw)
    result["raw_response"] = raw
    return result


def _parse_json(raw: str) -> dict:
    try:
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{.*?\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    texto = raw.lower()
    converged = any(
        w in texto for w in ["convergi", "equivalente", "igual", "similar", "mesmas"]
    )
    return {
        "converged":          converged,
        "reason":             "Parse do JSON falhou — resposta interpretada por heurística",
        "top_features_match": False,
        "direction_match":    False,
        "confidence":         0.3,
    }
