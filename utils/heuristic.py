import numpy as np

def normalize_transformer_logits(logits_k):
    """
    Normaliza os logits do Transformer para probabilidades usando a função softmax.

    Args:
        logits_k (list or np.array): Uma lista ou array numpy [logit_NV, logit_V]
                                      para uma instância k.

    Returns:
        np.array: Probabilidades normalizadas [P(NV), P(V)] para a instância k.
    """
    logits_k = np.array(logits_k) # Garante que é um array numpy para operações
    exp_logits = np.exp(logits_k - np.max(logits_k)) # Subtrai o máximo para estabilidade numérica
    probabilities = exp_logits / np.sum(exp_logits)
    return probabilities

def normalize_hmm_scores(scores_k):
    """
    Normaliza os scores de probabilidade do HMM para probabilidades.

    Args:
        scores_k (list or np.array): Uma lista ou array numpy [score_NV, score_V]
                                     para uma instância k.

    Returns:
        np.array: Probabilidades normalizadas [P(NV), P(V)] para a instância k.
    """
    scores_k = np.array(scores_k) # Garante que é um array numpy para operações
    total_score = np.sum(scores_k)

    if total_score == 0:
        # Lida com o caso em que ambos os scores são zero (evita divisão por zero)
        return np.array([0.5, 0.5]) # Ou outra distribuição padrão de sua escolha
    
    probabilities = scores_k / total_score
    return probabilities

def calculate_heuristic_confidence(p_hmm_k, p_trans_k, p_lstm_k, acc1_hmm, acc2_transformer, acc3_lstm):
    """
    Calcula a heurística de confiança para as classes, variando de 0 a 2.

    Args:
        p_hmm_k (np.array): Probabilidades HMM normalizadas [P(NV), P(V)] para a instância k.
        p_t_k (np.array): Probabilidades Transformer normalizadas [P(NV), P(V)] para a instância k.
        p_lstm_k (np.array): Probabilidades LSTM normalizadas [P(NV), P(V)] para a instância k.
        acc1_hmm (float): Acurácia do modelo HMM (entre 0 e 1).
        acc2_transformer (float): Acurácia do modelo Transformer (entre 0 e 1).
        acc3_lstm (float): Acurácia do modelo LSTM (entre 0 e 1).

    Returns:
        list: Uma lista com os scores de confiança [NV, V].
    """
    confidence_nv = acc1_hmm * p_hmm_k[0] + acc2_transformer * p_trans_k[0] + acc3_lstm * p_lstm_k[0]
    confidence_v = acc1_hmm * p_hmm_k[1] + acc2_transformer * p_trans_k[1] + acc3_lstm * p_lstm_k[1]

    return [confidence_nv, confidence_v]

def convert_to_percentage(confidence_score):
    """
    Converte um score de confiança (intervalo 0-3) para uma percentagem (0-100%).

    Args:
        confidence_score (float): O score de confiança calculado (entre 0 e 3).

    Returns:
        float: A confiança em percentagem (entre 0 e 100).
    """
    return round(confidence_score * 33.33, 2)  # Multiplica por 33.33 para converter de 0-3 para 0-100%




if __name__ == "__main__":
    # --- Exemplo de Uso Completo para uma Única Instância ---

    # 1. Dados de exemplo para uma instância 'k'
    # Seus logits do Transformer para NV e V
    logits_transformer_k = [-0.17742497, 0.85725254]

    # Seus scores de probabilidade do HMM para NV e V
    # Lembre-se, esses são os valores originais que você forneceu,
    # antes da normalização do HMM.
    scores_hmm_k = [5.779428797388737e-28, 4.444206140965821e-27]

    # Seus logits do LSTM para NV e V
    logits_lstm_k = [0.2, 0.8]  # Exemplo de logits do LSTM
    # Normalizar logits do LSTM para probabilidades
    probabilities_lstm_k = normalize_transformer_logits(logits_lstm_k)

    # Acurácias dos seus modelos (exemplo)
    accuracy_hmm = 0.85 # Exemplo: 85% de acurácia para o HMM
    accuracy_transformer = 0.92 # Exemplo: 92% de acurácia para o Transformer
    accuracy_lstm = 0.88 # Exemplo: 88% de acurácia para o LSTM

    print(f"--- Calculando Confiança para Instância K ---")

    # 2. Normalizar Logits do Transformer
    probabilities_transformer_k = normalize_transformer_logits(logits_transformer_k)
    print(f"Probabilidades Transformer (P_t): NV={probabilities_transformer_k[0]:.4f}, V={probabilities_transformer_k[1]:.4f}")

    # 3. Normalizar Scores do HMM
    probabilities_hmm_k = normalize_hmm_scores(scores_hmm_k)
    print(f"Probabilidades HMM (P_hmm): NV={probabilities_hmm_k[0]:.4f}, V={probabilities_hmm_k[1]:.4f}")

    # 4. Calcular a Heurística de Confiança (0-2)
    heuristic_scores_k = calculate_heuristic_confidence(
        probabilities_hmm_k,
        probabilities_transformer_k,
        probabilities_lstm_k,
        accuracy_hmm,
        accuracy_transformer,
        accuracy_lstm
    )
    print(f"Scores Heurísticos (0-2): NV={heuristic_scores_k[0]:.4f}, V={heuristic_scores_k[1]:.4f}")

    # 5. Converter para Confiança Percentual (0-100%)
    confidence_percent_nv = convert_to_percentage(heuristic_scores_k[0])
    confidence_percent_v = convert_to_percentage(heuristic_scores_k[1])
    print(f"Confiança Percentual: NV={confidence_percent_nv:.2f}%, V={confidence_percent_v:.2f}%")