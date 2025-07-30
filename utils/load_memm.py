import json
import re
import os
from collections import defaultdict
import numpy as np
import joblib 
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .viterbi import viterbi_MEMM


# --- Funções Auxiliares (usadas para processamento de dados e reestruturação) ---

def add_file_to_mappings(mappings: dict, filename: str, lines: list) -> dict:
    """
    Adiciona o mapeamento de tokens para IDs originais de um ficheiro ao dicionário global de mapeamentos.
    Dígitos nos tokens são preservados, outros são substituídos por "_".
    """
    file_map = []

    for line in lines:
        mapped_tokens = []
        for token in line:
            if re.search(r'\d', token):
                mapped_tokens.append(token)
            else:
                mapped_tokens.append("_")
        file_map.append(mapped_tokens)

    file_id = str(len(mappings) + 1)
    mappings[file_id] = {
        "nome": filename,
        "map": file_map
    }
    return mappings

def remove_var_no(lines):
    """
    Remove sufixos numéricos de construções de programação comuns como varN,
    function_nameN, class_nameN.
    Espera que 'lines' seja uma lista plana de tokens.
    """
    new_lines = []
    for line in lines:
        line = re.sub(r'\bvar\d+(?:_\d+)?\b', 'var', line)
        line = re.sub(r'\bfunction_name\d+\b', 'function_name', line)
        line = re.sub(r'\bclass_name\d+\b', 'class_name', line)
        line = re.sub(r'->function_name\d+\b', '->function_name', line)
        new_lines.append(line)
    return new_lines

def load_il_observation_sequences(filepath, mappings):
    """
    Carrega sequências de observação de um ficheiro Intermediate Language (IL).
    Também popula o dicionário 'mappings' com os IDs originais dos tokens.
    Retorna o nome do ficheiro e a sequência de observação aplainada e limpa.
    """
    obs_seq_raw_lines = []
    filename = os.path.basename(filepath)

    with open(filepath, "r") as f:
        for line in f:
            tokens = line.strip().split()
            obs_seq_raw_lines.append(tokens) # Preserva a linha como lista de tokens para mapeamento

    mappings = add_file_to_mappings(mappings, filename, obs_seq_raw_lines)

    if obs_seq_raw_lines:
        # Aplaina e inverte para o processamento MEMM (conforme a sua lógica original)
        flattened = [token for line in obs_seq_raw_lines for token in line[::-1]]
        cleaned = remove_var_no(flattened)
        return (filename, cleaned)
    else:
        return (filename, [])

def restructure_predictions_by_mapping(pred_tokens, pred_tags, mapping_lines):
    """
    Reestrutura tokens e tags preditos (aplainados) de volta em linhas,
    invertendo cada linha com base na estrutura de mapeamento original.
    """
    structured_tokens = []
    structured_tags = []

    idx = 0
    for line_map_entry in mapping_lines:
        line_len = len(line_map_entry) # Usa a entrada do mapa para determinar o comprimento original da linha
        line_tokens = pred_tokens[idx:idx + line_len]
        line_tags = pred_tags[idx:idx + line_len]

        # Inverte de volta para a ordem original para cada linha
        structured_tokens.append(list(reversed(line_tokens)))
        structured_tags.append(list(reversed(line_tags)))

        idx += line_len

    return structured_tokens, structured_tags

def reconstruct_with_ids_by_line(pred_filename, tokens_by_line, tags_by_line, mappings):
    """
    Reconstrói os tokens originais (com seus IDs, se presentes)
    e os emparelha com as tags preditas, linha por linha.
    """
    mapping_entry = next((v for v in mappings.values() if v["nome"] == pred_filename), None)
    if not mapping_entry:
        raise ValueError(f"Filename {pred_filename} not found in mappings.")

    full_map = mapping_entry["map"]

    reconstructed = []
    for map_line, token_line, tag_line in zip(full_map, tokens_by_line, tags_by_line):
        if not (len(map_line) == len(token_line) == len(tag_line)):
            raise ValueError("Line lengths don't match during reconstruction.")

        line_result = []
        for token_map, token_pred, tag_pred in zip(map_line, token_line, tag_line):
            if token_map == "_": # Se era um token genérico, usa o token predito
                line_result.append((token_pred, tag_pred))
            else: # Se tinha um ID específico, usa o token original
                line_result.append((token_map, tag_pred))
        reconstructed.append(line_result)

    return reconstructed


def load_memm(model_path, input_filepath, verbose=False):
    """
    Carrega um modelo MEMM treinado e usa-o para prever tags para um dado ficheiro de entrada.

    Args:
        model_path (str): O caminho para o modelo MEMM treinado (.pkl).
        input_filepath (str): O caminho para o ficheiro de entrada contendo sequências de observação.
        verbose (bool, optional): Se True, imprime as tags preditas para cada linha. Padrão para False.

    Returns:
        list: Uma lista de listas, onde cada lista interna representa uma linha do ficheiro de entrada,
              e contém tuplos de (token, tag_predita). Retorna uma lista vazia se o modelo não puder
              ser carregado ou se nenhuma observação for encontrada.
    """
    # 1. Carregar o modelo MEMM
    try:
        modelo_MEMM = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Erro: Modelo MEMM não encontrado em '{model_path}'. Por favor, verifique o caminho.")
        return []
    except Exception as e:
        print(f"Erro ao carregar o modelo MEMM de '{model_path}': {e}")
        return []

    # 2. Obter as etiquetas (tags) possíveis do modelo carregado
    # Assumindo que o último passo do pipeline é um classificador com um atributo 'classes_'.
    try:
        etiquetas_possiveis = sorted(list(modelo_MEMM.named_steps['classifier'].classes_))
        # Garantir que 'START' não está nas etiquetas de predição se for apenas um estado inicial do Viterbi
        if 'START' in etiquetas_possiveis:
             etiquetas_possiveis.remove('START')
    except AttributeError:
        print("Aviso: Não foi possível inferir 'etiquetas_possiveis' do modelo carregado. "
              "O Viterbi pode falhar ou produzir resultados incorretos se as etiquetas forem desconhecidas. "
              "Por favor, certifique-se de que o classificador do seu modelo tem um atributo 'classes_'.")
        return [] # Não pode prosseguir sem etiquetas

    # 3. Processar o ficheiro de entrada para obter sequências de observação
    mappings = {} # Inicializa os mapeamentos para este ficheiro
    filename, obs_seq = load_il_observation_sequences(input_filepath, mappings)

    # Se a sequência de observação estiver vazia após o processamento, retorna um resultado vazio
    if not obs_seq:
        if verbose:
            print(f"Nenhuma observação válida encontrada em '{input_filepath}'. Pulando a predição.")
        return []

    # 4. Executar inferência Viterbi usando o modelo MEMM
    pred_tags = viterbi_MEMM(modelo_MEMM, obs_seq, etiquetas_possiveis)

    # Lidar com casos em que pred_tags pode estar vazio (ex: se obs_seq estava vazio inicialmente)
    if not pred_tags:
        if verbose:
            print(f"Viterbi não retornou predições para '{filename}'.")
        return []

    # 5. Reestruturar as predições com base no mapeamento original do ficheiro
    # Obtém o mapeamento para o ficheiro atual (load_il_observation_sequences popula 'mappings')
    mapping_lines = next(v["map"] for v in mappings.values() if v["nome"] == filename)

    # Reestrutura os tokens e tags aplainados de volta à sua estrutura de linha original
    tokens_by_line, tags_by_line = restructure_predictions_by_mapping(obs_seq, pred_tags, mapping_lines)

    # 6. Reconstruir a saída com os IDs originais e as tags preditas
    final_reconstruction = reconstruct_with_ids_by_line(filename, tokens_by_line, tags_by_line, mappings)

    if verbose:
        print(f"\nFicheiro: {filename}")
        for line in final_reconstruction:
            print(" ".join(f"{token}::{tag}" for token, tag in line))

    return final_reconstruction