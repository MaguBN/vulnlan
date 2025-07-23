from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def load_transformer_model(model_path, verbose=True):
    """
    Load a pre-trained transformer model and its tokenizer from the specified path.
    
    Args:
        model_path (str): Path to the directory containing the pre-trained model.
        
    Returns:
        model: The loaded transformer model.
        tokenizer: The loaded tokenizer for the model.
    """
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if verbose:
        print("Tokenizador carregado com sucesso.")
    
    
    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    if verbose:
        print("Modelo carregado com sucesso.")

    # Move the model to the appropriate device and set it to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if verbose:
        print("Modelo e tokenizador prontos para inferência.")

    return model, tokenizer
