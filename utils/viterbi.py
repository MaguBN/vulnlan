import math
import numpy as np

SMOOTH = 1e-6

def viterbi(sequence, states, start_probs, trans_probs, emit_probs, smooth=SMOOTH):
    """
    Algoritmo de Viterbi para HMMs em espaço logarítmico.

    sequence: lista de observações (strings)
    states: conjunto de estados possíveis
    start_probs: dicionário de probabilidades iniciais
    trans_probs: defaultdict com probabilidades de transição
    emit_probs: defaultdict com probabilidades de emissão
    smooth: valor para smoothing (probabilidade mínima)

    Retorna:
        best_path: lista com a sequência de estados mais provável
        max_log_prob: log da probabilidade do caminho mais provável
    """
    V = [{}]
    path = {}

    # Inicialização
    for state in states:
        start_p = start_probs.get(state, smooth)
        emit_p = emit_probs.get(state, {}).get(sequence[0], smooth)
        V[0][state] = math.log(max(start_p, smooth)) + math.log(max(emit_p, smooth))
        path[state] = [state]

    # Iteração dinâmica
    for t in range(1, len(sequence)):
        V.append({})
        new_path = {}

        for curr_state in states:
            emit_p = emit_probs.get(curr_state, {}).get(sequence[t], smooth)

            max_prob, prev_state = max(
                (
                    V[t - 1][prev_state]
                    + math.log(max(trans_probs.get(prev_state, {}).get(curr_state, smooth), smooth))
                    + math.log(max(emit_p, smooth)),
                    prev_state,
                )
                for prev_state in states
            )

            V[t][curr_state] = max_prob
            new_path[curr_state] = path[prev_state] + [curr_state]

        path = new_path

    # Finalização
    final_state = max(V[-1], key=V[-1].get)
    return path[final_state], V[-1][final_state]


def probs_MEMM(modelo, observacao, estado_anterior, etiquetas):
    features = {
        'observacao': observacao,
        'estado_anterior': estado_anterior
    }
    proba = modelo.predict_proba([features])[0]
    etiqueta_proba = dict(zip(modelo.classes_, proba))
    return [etiqueta_proba.get(et, 1e-6) for et in etiquetas]


def viterbi_MEMM(modelo, observacoes, etiquetas):
    T = len(observacoes)
    N = len(etiquetas)
    dp = np.zeros((T, N))
    backpointer = np.zeros((T, N), dtype=int)

    # Handle potential empty observacoes list
    if T == 0:
        return []

    # Initialize first step
    for j, s in enumerate(etiquetas):
        dp[0][j] = np.log(probs_MEMM(modelo, observacoes[0], 'START', etiquetas)[j])

    # Fill DP table
    for t in range(1, T):
        for j, s in enumerate(etiquetas):
            max_prob = float('-inf')
            max_state = 0
            for i, s_prev in enumerate(etiquetas):
                prob = dp[t - 1][i] + np.log(probs_MEMM(modelo, observacoes[t], s_prev, etiquetas)[j])
                if prob > max_prob:
                    max_prob = prob
                    max_state = i
            dp[t][j] = max_prob
            backpointer[t][j] = max_state

    # Backtrack
    best_path = []
    last_state = np.argmax(dp[-1])
    best_path.append(etiquetas[last_state])
    for t in range(T - 1, 0, -1):
        last_state = backpointer[t][last_state]
        best_path.insert(0, etiquetas[last_state])

    return best_path