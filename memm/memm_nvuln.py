import os
import numpy as np
import math
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def carregar_ficheiros_MEMM(pasta):
    sequencias = []
    # Check if the directory exists
    if not os.path.exists(pasta):
        print(f"Erro: A pasta '{pasta}' não existe.")
        return sequencias
    if not os.listdir(pasta):
        print(f"A pasta '{pasta}' está vazia.")
        return sequencias

    print(f"A procurar ficheiros '.ann' na pasta: {pasta}")
    found_files_count = 0
    for nome_ficheiro in os.listdir(pasta):
        if nome_ficheiro.endswith('.ann'):
            found_files_count += 1
            caminho = os.path.join(pasta, nome_ficheiro)
            tokens_estado = []
            try:
                with open(caminho, 'r', encoding='utf-8') as f:
                    for linha in f:
                        tokens = linha.strip().split()
                        # print(f"    Linha: '{linha.strip()}', Tokens: {tokens}") # Debug: See raw tokens
                        for token in tokens[::-1]:
                            if "::" in token:
                                partes = token.split("::")
                                if len(partes) == 2:
                                    obs, estado = partes
                                    tokens_estado.append((obs, estado))
                                # else:
                                    # print(f"      Token '{token}' has '::' but not 2 parts after split.") # Debug: Check malformed tokens
                            # else:
                                # print(f"      Token '{token}' does not contain '::'.") # Debug: See tokens without '::'
                    if tokens_estado:
                        sequencias.append(tokens_estado)
                    # else:
                        # print(f"    Ficheiro '{nome_ficheiro}' não gerou tokens_estado com '::'.") # Debug: File did not yield any valid pairs
            except Exception as e:
                print(f"    Erro ao ler o ficheiro {nome_ficheiro}: {e}")
    print(f"Total de ficheiros .ann encontrados: {found_files_count}")
    print(f"Total de sequências carregadas com sucesso: {len(sequencias)}")
    return sequencias

def extrair_features_MEMM(sequencias):
    X, y = [], []
    print(f"A extrair features de {len(sequencias)} sequências...")
    for i, seq in enumerate(sequencias):
        estado_anterior = "START"
        for observacao, estado in seq:
            features = {
                'observacao': observacao,
                'estado_anterior': estado_anterior
            }
            X.append(features)
            y.append(estado)
            estado_anterior = estado
        # print(f"  Sequência {i}: {len(seq)} pares processados. Total X agora: {len(X)}") # Debug: Track progress per sequence
    print(f"Extração de features concluída. Total de amostras X: {len(X)}")
    return X, y

def treinar_MEMM(X, y):
    print(f"A treinar o modelo com {len(X)} amostras...")
    modelo = Pipeline([
        ('vectorizer', DictVectorizer(sparse=False)),
        ('classifier', LogisticRegression(max_iter=300, class_weight='balanced'))
    ])
    modelo.fit(X, y)
    print("Modelo treinado com sucesso.")
    return modelo

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

def split_annotated_files(files, test_size=0.15):
    train_files, test_files = train_test_split(files, test_size=test_size, random_state=31)
    return train_files, test_files

# --- Execução Principal ---
pasta_dados_vuln = "../data/ann_ss/nvuln"

# Carrega os dados
sequencias = carregar_ficheiros_MEMM(pasta_dados_vuln)

train_seqs, test_seqs = split_annotated_files(sequencias)

# 1. Extrair features e labels APENAS do conjunto de TREINO
print("\nExtraindo features e labels do conjunto de TREINO...")
X_train, y_train = extrair_features_MEMM(train_seqs)

if not X_train:
    print("Erro: X_train está vazio. Não é possível treinar o modelo.")
else:
    # 2. Treinar o modelo APENAS com X_train e y_train
    print("Treinando o modelo MEMM...")
    modelo_MEMM = treinar_MEMM(X_train, y_train)

    # Obter todas as etiquetas possíveis do conjunto de treino (para garantir que o Viterbi as conheça)
    etiquetas_possiveis = sorted(list(set(y_train)))
    print(f"Etiquetas possíveis encontradas no conjunto de treino: {etiquetas_possiveis}")

    # --- Avaliação no conjunto de TESTE ---
    print("\nAvaliação do modelo no conjunto de TESTE...")
    all_true_labels = []
    all_predicted_labels = []

    for test_sequence in test_seqs:
        # Extract observations for the current test sequence
        test_obs_current_seq = [obs for obs, state in test_sequence]
        true_states_current_seq = [state for obs, state in test_sequence]

        if test_obs_current_seq: # Only predict if there are observations
            predicted_states_current_seq = viterbi_MEMM(modelo_MEMM, test_obs_current_seq, etiquetas_possiveis)
            all_true_labels.extend(true_states_current_seq)
            all_predicted_labels.extend(predicted_states_current_seq)


    if all_true_labels and all_predicted_labels:
        # Ensure that both lists have the same length
        min_len = min(len(all_true_labels), len(all_predicted_labels))
        all_true_labels = all_true_labels[:min_len]
        all_predicted_labels = all_predicted_labels[:min_len]

        print("\nRelatório de Classificação no conjunto de TESTE:")
        print(classification_report(all_true_labels, all_predicted_labels, zero_division=0, digits=3))

    # Save the trained model
    joblib.dump(modelo_MEMM, "modelo_MEMM_nvuln.pkl")
