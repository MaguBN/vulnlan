import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import joblib
# Verbose
from utils.verbose import verbose
# Para traduzir
from PHP2IL.PHPDecomposer import PHPDecomposer
from PHP2IL.main import PHP2IL
# Modelo LSTM
from utils.load_lstm import load_lstm, encode_snippet_for_inference
# Modelo Transformer
from utils.load_trans import load_transformer_model
# Modelo HMM unsupervised
from utils.load_hmm_unsup import load_hmm_from_json, prever_sql_injection
# Heuristica
from utils.heuristic import calculate_heuristic_confidence, convert_to_percentage
# Modelo HMM supervised
from utils.load_hmm_sup import load_hmm, load_il_observation_sequences, remove_var_no, restructure_predictions_by_mapping, reconstruct_with_ids_by_line

def get_lstm_prediction(model, input_ids):
    with torch.no_grad():
        logits = model(input_ids)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
    return logits, probabilities, predicted_class

def get_transformer_prediction(model, inputs_dict):
    device = next(model.parameters()).device
    inputs_on_device = {key: value.to(device) for key, value in inputs_dict.items()}
    with torch.no_grad():
        outputs = model(**inputs_on_device)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(logits, dim=1).item()
    return logits, probabilities, predicted_class

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Traduzir de PHP para IL
    #PHP2IL(['-s', 'input_files/php/', '-d', 'input_files/il/'])


    # Load snippets in IL from folder 'input_files/il/'
    il_snippets_with_names = []
    il_folder = 'input_files/il/'
    filenames = os.listdir(il_folder)
    for filename in filenames:
        if filename.endswith('.il'):
            with open(os.path.join(il_folder, filename), 'r') as file:
                snippet = file.read().strip()
                il_snippets_with_names.append((filename, snippet))

    # Prepare a list to store info for every snippet
    snippets_info = []
    for snippet_name, snippet_content in il_snippets_with_names:
        snippet_info = {
            'snippet_name': snippet_name,
            'snippet_content': snippet_content,
            'predicted_class': 0,
            'lstm_probs': [],
            'transformer_probs': [],
            'hmm_probs': [],
            'heuristic_confidence_nv': 0,
            'heuristic_confidence_v': 0
        }
        snippets_info.append(snippet_info)

    # Load LSTM
    lstm_model, lstm_vocab, lstm_max_len = load_lstm("lstm/lstm_model.pth", verbose=False)
    # LSTM inference
    for snippet in snippets_info:
        encoded_snippet = encode_snippet_for_inference(snippet['snippet_content'], lstm_vocab, lstm_max_len)
        lstm_input_ids = torch.tensor(encoded_snippet).unsqueeze(0)
        
        lstm_logits, lstm_probs_tensor, lstm_predicted_class = get_lstm_prediction(lstm_model, lstm_input_ids)

        # Converter para lista de floats
        lstm_probs = lstm_probs_tensor.detach().cpu().numpy().flatten().tolist()

        snippet['lstm_probs'] = lstm_probs
        snippet['predicted_class'] = lstm_predicted_class

        # Chama verbose usando snippet_name e snippet_content
        verbose("LSTM", snippet['snippet_name'], snippet['snippet_content'], snippet['predicted_class'], snippet['lstm_probs'], show_snippet=False)
    
    # Load Transformers
    model_path = "transformers/with_id/transformer_model_id/"
    loaded_model, loaded_tokenizer = load_transformer_model(model_path, verbose=False)

    # Transformer inference
    for snippet in snippets_info:
        inputs = loaded_tokenizer(snippet['snippet_content'], return_tensors="pt", padding=True, truncation=True)
        
        logits, probs_tensor, pred_class = get_transformer_prediction(loaded_model, inputs)

        # Converter para lista de floats no CPU
        probs = probs_tensor.detach().cpu().numpy().flatten().tolist()

        snippet['transformer_probs'] = probs
        snippet['predicted_class'] = pred_class

        # Pass the already available snippet_name and snippet_content to verbose
        verbose("Transformer", snippet['snippet_name'], snippet['snippet_content'], snippet['predicted_class'], snippet['transformer_probs'], show_snippet=False)


    # Load HMM unsupervised
    modelo_vuln = load_hmm_from_json("hmm/hmm_unsup_vuln.json")
    modelo_nao_vuln = load_hmm_from_json("hmm/hmm_unsup_nvuln.json")

    # Load encoder (deve ser o mesmo usado no treino!)
    encoder = joblib.load("hmm/ordinal_encoder.pkl")  # ou .pkl/.joblib salvo antes com joblib.dump(encoder, ...)

    # HMM inference
    for snippet in snippets_info:
        snippet['predicted_class'], snippet['hmm_probs'] = prever_sql_injection(snippet['snippet_content'], modelo_vuln, modelo_nao_vuln, encoder)
        verbose("HMM unsupervised", snippet['snippet_name'], snippet['snippet_content'], snippet['predicted_class'], snippet['hmm_probs'], show_snippet=False)

    # Heuristic confidence calculation
    acc1_hmm = 0.7
    acc2_transformer = 0.95
    acc3_lstm = 0.93

    for snippet in snippets_info:
        # Obter as probabilidades do HMM, Transformer e LSTM
        p_hmm_k = snippet['hmm_probs']
        p_trans_k = snippet['transformer_probs']
        p_lstm_k = snippet['lstm_probs']

        # Calcular a confiança usando a heurística
        confidence_scores = calculate_heuristic_confidence(p_hmm_k, p_trans_k, p_lstm_k, acc1_hmm, acc2_transformer, acc3_lstm)
        snippet['heuristic_confidence_nv'] = round(confidence_scores[0], 2)
        snippet['heuristic_confidence_v'] = round(confidence_scores[1], 2)

    # Exibir resultados
    for snippet in snippets_info:
        if snippet['heuristic_confidence_nv'] > snippet['heuristic_confidence_v']:
            snippet['predicted_class'] = 0
        else: 
            snippet['predicted_class'] = 1
        print(f"Snippet: {snippet['snippet_name']}")
        print(f"  Heuristic Confidence (Non-Vulnerable): {snippet['heuristic_confidence_nv']} ({convert_to_percentage(snippet['heuristic_confidence_nv'])}%)")
        print(f"  Heuristic Confidence (Vulnerable): {snippet['heuristic_confidence_v']} ({convert_to_percentage(snippet['heuristic_confidence_v'])}%)")

    # CMD prompt for continuing with the next steps
    print("\nENTER to continue with the labeling of the snippets with the HMM and MEMM supervised model...")
    input()

    # Load HMM supervised model
    for snippet in snippets_info:
        model_path = "hmm/hmm_sup_vuln.json" if snippet['predicted_class'] == 1 else "hmm/hmm_sup_nvuln.json"
        hmm_model = load_hmm(model_path, f"input_files/il/{snippet['snippet_name']}", verbose=True)
        
