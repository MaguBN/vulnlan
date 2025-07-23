import torch
import torch.nn as nn
# Importe outras bibliotecas que precise para inferência (numpy, etc.)
# import numpy as np # Se precisar de operações com arrays

# --- Definição da Classe do Modelo (precisa ser EXATAMENTE a mesma) ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim, dropout):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        final_output = lstm_out[:, -1, :] 
        out = self.fc(final_output)
        return out

# --- Função de Encoding (precisa ser a mesma, e usará o max_len carregado) ---
def encode_snippet_for_inference(snippet, vocab_map, max_len):
    tokens = snippet.split()
    token_ids = [vocab_map.get(token, 0) for token in tokens] # '0' para UNK
    if len(token_ids) > max_len:
        return token_ids[:max_len]
    else:
        return token_ids + [0] * (max_len - len(token_ids))

def load_lstm(model_path, verbose=True):

    # Carregar o checkpoint completo
    checkpoint = torch.load(model_path, map_location=torch.device('cpu')) # map_location para carregar na CPU se treinou na GPU

    # Extrair o vocabulário e os hiperparâmetros
    loaded_vocab = checkpoint['vocab']
    loaded_hyperparameters = checkpoint['hyperparameters']

    # Reconstruir os hiperparâmetros para instanciar o modelo
    vocab_size = loaded_hyperparameters['vocab_size']
    embedding_dim = loaded_hyperparameters['embedding_dim']
    hidden_dim = loaded_hyperparameters['hidden_dim']
    num_layers = loaded_hyperparameters['num_layers']
    output_dim = loaded_hyperparameters['output_dim']
    dropout = loaded_hyperparameters['dropout']
    max_len = loaded_hyperparameters['max_len'] # Este é o max_len usado para encoding

    # Instanciar o modelo
    loaded_model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, num_layers, output_dim, dropout)

    # Carregar os pesos (state_dict) para o modelo instanciado
    loaded_model.load_state_dict(checkpoint['model_state_dict'])

    # Mover o modelo para o dispositivo e colocar em modo de avaliação
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # loaded_model.to(device)
    loaded_model.eval() # Coloca o modelo em modo de avaliação (desativa dropout e usa estatísticas de batch norm para inferência)

    if verbose:
        print(f"Modelo carregado com sucesso de: {model_path}")
        print(f"Vocabulário carregado com {len(loaded_vocab)} tokens.")
        print(f"Hiperparâmetros carregados: {loaded_hyperparameters}")

    return loaded_model, loaded_vocab, max_len
