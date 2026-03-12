import requests


SLM_URL   = "http://localhost:12434/engines/llama.cpp/v1/chat/completions"
SLM_MODEL = "ai/qwen2.5:7B-Q4_0"
TIMEOUT   = 120

FEATURE_NAMES_PT = {
    "loan_percent_income":        "proporção do empréstimo em relação à sua renda",
    "person_income":              "sua renda",
    "person_home_ownership":      "tipo de moradia",
    "loan_intent":                "finalidade do empréstimo",
    "loan_int_rate":              "taxa de juros",
    "loan_amnt":                  "valor do empréstimo",
    "loan_grade":                 "classificação do empréstimo",
    "person_age":                 "sua idade",
    "person_emp_length":          "tempo de emprego",
    "cb_person_default_on_file":  "histórico de inadimplência",
    "cb_person_cred_hist_length": "tempo de histórico de crédito",
}


def _traduzir_feature(feature_str: str) -> str:
    for nome, traducao in FEATURE_NAMES_PT.items():
        if nome in feature_str:
            return traducao
    return feature_str.replace("_", " ")


def _formatar_features(lime_features: list) -> str:
    linhas = []
    for feature_str, weight in lime_features:
        direcao = "FAVORECE" if weight > 0 else "PREJUDICA"
        abs_weight = abs(weight)
        if abs_weight >= 0.20:
            intensidade = "muito"
        elif abs_weight >= 0.10:
            intensidade = "moderadamente"
        else:
            intensidade = "levemente"
        traducao = _traduzir_feature(feature_str)
        linhas.append(f"- {traducao}: {intensidade} {direcao} a adimplência")
    return "\n".join(linhas)


def generate_explanation(
    prediction: str,
    probability: float,
    lime_features: list,
    context: str = "sistema de análise de crédito bancário",
) -> str:
    features_str = _formatar_features(lime_features)

    prompt = f"""Você é um assistente que explica decisões de {context} para clientes comuns, em português do Brasil.

O sistema avaliou este cliente:
- Decisão: {prediction}
- Probabilidade de inadimplência: {probability * 100:.1f}%
- Probabilidade de adimplência: {(1 - probability) * 100:.1f}%

Fatores que influenciaram a decisão:
{features_str}

Escreva um parágrafo de 3 a 5 frases explicando a decisão diretamente para o cliente.

Regras obrigatórias:
- Use "você" e "sua" para se dirigir ao cliente
- NÃO use: LIME, feature, peso, coeficiente, modelo, algoritmo
- Explique de forma causal e humana
- Seja empático e construtivo
- Escreva apenas o parágrafo, sem títulos ou listas"""

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

    return response.json()["choices"][0]["message"]["content"].strip()

