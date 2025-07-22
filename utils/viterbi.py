import math

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