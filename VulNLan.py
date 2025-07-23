import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
# Verbose
from utils.verbose import verbose
# Para traduzir
from PHP2IL.PHPDecomposer import PHPDecomposer
from PHP2IL.main import PHP2IL
# Modelo LSTM
from utils.load_lstm import load_lstm, encode_snippet_for_inference
# Modelo Transformer
from utils.load_trans import load_transformer_model


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


    # Load LSTM
    lstm_model, lstm_vocab, lstm_max_len = load_lstm("lstm/lstm_model.pth", verbose=False)
    # LSTM inference
    for snippet_name, snippet_content in il_snippets_with_names:
        encoded_snippet = encode_snippet_for_inference(snippet_content, lstm_vocab, lstm_max_len)
        lstm_input_ids = torch.tensor(encoded_snippet).unsqueeze(0)
        
        # Chama a função 'get_lstm_prediction'
        lstm_logits, lstm_probabilities, lstm_predicted_class = get_lstm_prediction(lstm_model, lstm_input_ids)
        
        # Chama verbose usando snippet_name e snippet_content
        verbose("LSTM", snippet_name, snippet_content, lstm_predicted_class, lstm_probabilities)


    # Load Transformers
    model_path = "transformers/with_id/transformer_model_id/"
    loaded_model, loaded_tokenizer = load_transformer_model(model_path, verbose=False)

    # Transformer inference
    for snippet_name, snippet_content in il_snippets_with_names: 
        inputs = loaded_tokenizer(snippet_content, return_tensors="pt", padding=True, truncation=True)
        
        logits, probabilities, predicted_class = get_transformer_prediction(loaded_model, inputs)
        
        # Pass the already available snippet_name and snippet_content to verbose
        verbose("Transformer", snippet_name, snippet_content, predicted_class, probabilities)