import json
from collections import defaultdict
from sklearn.metrics import classification_report
from .viterbi import viterbi
import os
import random
import re

def load_hmm_model(model_path):
    with open(model_path, "r") as f:
        model_dict = json.load(f)

    # Reconstruir defaultdicts
    start_probs = model_dict["start_probs"]
    trans_probs = defaultdict(dict, model_dict["trans_probs"])
    emit_probs = defaultdict(dict, model_dict["emit_probs"])
    states = set(model_dict["states"])

    return start_probs, trans_probs, emit_probs, states

def add_file_to_mappings(mappings: dict, filename: str, lines: list) -> dict:
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
    new_lines = []
    for line in lines:
        
        line = re.sub(r'\bvar\d+(?:_\d+)?\b', 'var', line)

        line = re.sub(r'\bfunction_name\d+\b', 'function_name', line)
        
        line = re.sub(r'\bclass_name\d+\b', 'class_name', line)
        
        line = re.sub(r'->function_name\d+\b', '->function_name', line)

        new_lines.append(line)

    return new_lines

def load_il_observation_sequences(filepath, mappings):
    obs_seq = []

    filename = os.path.basename(filepath)

    with open(filepath, "r") as f:
        for line in f:
            tokens = line.strip().split()
            obs_seq.append(tokens)  # ← preservar linha como lista de tokens

    mappings = add_file_to_mappings(mappings, filename, obs_seq)

    if obs_seq:
        # Flatten para o modelo HMM
        flattened = [token for line in obs_seq for token in line[::-1]]
        cleaned = remove_var_no(flattened)
        return (filename, cleaned)
    else:
        return (filename, [])

def restructure_predictions_by_mapping(pred_tokens, pred_tags, mapping_lines):
    """
    Divide e inverte tokens e tags por linha com base no mapping.
    """
    structured_tokens = []
    structured_tags = []
    
    idx = 0
    for line in mapping_lines:
        line_len = len(line)
        # Garantir que não estamos a exceder o comprimento
        line_tokens = pred_tokens[idx:idx + line_len]
        line_tags = pred_tags[idx:idx + line_len]

        structured_tokens.append(list(reversed(line_tokens)))
        structured_tags.append(list(reversed(line_tags)))

        idx += line_len

    return structured_tokens, structured_tags

def reconstruct_with_ids_by_line(pred_filename, tokens_by_line, tags_by_line, mappings):
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
            if token_map == "_":
                line_result.append((token_pred, tag_pred))
            else:
                line_result.append((token_map, tag_pred))
        reconstructed.append(line_result)
    
    return reconstructed

def load_hmm(model_path, input_filepath, verbose=False):
    # Carregar modelo HMM
    start_probs, trans_probs, emit_probs, states = load_hmm_model(model_path)

    mappings = {}

    # Processar apenas 1 ficheiro
    filename, obs_seq = load_il_observation_sequences(input_filepath, mappings)

    pred_tags, _ = viterbi(obs_seq, states, start_probs, trans_probs, emit_probs)

    mapping_lines = next(v["map"] for v in mappings.values() if v["nome"] == filename)
    tokens_by_line, tags_by_line = restructure_predictions_by_mapping(obs_seq, pred_tags, mapping_lines)
    final = reconstruct_with_ids_by_line(filename, tokens_by_line, tags_by_line, mappings)

    if verbose:
        print(f"\nFicheiro: {filename}")
        for line in final:
            print(" ".join(f"{token}::{tag} " for token, tag in line))

    return final  # ← útil se quiseres usar o resultado noutro lado