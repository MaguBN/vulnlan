import json
import numpy as np
from hmmlearn import hmm

def load_hmm_from_json(filepath):
    with open(filepath, "r") as f:
        model_data = json.load(f)

    n_components = len(model_data["start_probs"])
    n_features = len(model_data["emit_probs"][0])

    model = hmm.CategoricalHMM(n_components=n_components, n_iter=100)
    model.startprob_ = np.array(model_data["start_probs"])
    model.transmat_ = np.array(model_data["trans_probs"])
    model.emissionprob_ = np.array(model_data["emit_probs"])
    model.n_features = n_features  # importante para evitar erros internos

    return model

def prever_sql_injection(snippet, modelo_vuln, modelo_nao_vuln, encoder):
    # Tokenizar e inverter
    tokens = snippet.strip().split()[::-1]

    # Codificar os tokens
    arr = np.array(tokens).reshape(-1, 1)
    try:
        seq = encoder.transform(arr).astype(int)
    except:
        seq = encoder.transform(arr)

    # Filtrar tokens fora do vocabulário (por segurança)
    max_index = modelo_vuln.emissionprob_.shape[1]
    seq = seq[(seq >= 0) & (seq < max_index)].reshape(-1, 1)

    if len(seq) == 0:
        print("[AVISO] Nenhum token válido fornecido.")
        return 0, [1.0, 0.0]  # assume não vulnerável

    # Calcular os scores dos HMMs
    score_vuln = modelo_vuln.score(seq)
    score_nao_vuln = modelo_nao_vuln.score(seq)

    # Softmax para normalizar os log-likelihoods
    prob_vuln = np.exp(score_vuln)
    prob_nao_vuln = np.exp(score_nao_vuln)
    total = prob_vuln + prob_nao_vuln
    prob_vuln /= total
    prob_nao_vuln /= total

    predicted_class = 1 if prob_vuln > prob_nao_vuln else 0
    probabilities = [prob_nao_vuln, prob_vuln]  # [não vulnerável, vulnerável]

    return predicted_class, probabilities