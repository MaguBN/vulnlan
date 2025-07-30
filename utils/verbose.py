import torch

def verbose(model_type, snippet_name, snippet_content, prediction, probabilities, show_snippet=True):

    print(f"\nAnálise ({model_type}) de: {snippet_name}") # Added model_type here
    if show_snippet:
        print(f"  Snippet:\n{snippet_content}\n") # Renamed from 'snippet' for clarity
    print(f"  Previsão: {'Vulnerável' if prediction == 1 else 'Não Vulnerável'}")
    # --- PROBABILITIES SIMPLIFICATION STARTS HERE ---
    if isinstance(probabilities, torch.Tensor):
        # Move tensor to CPU and convert to a NumPy array, then to a list
        # We assume probabilities are in the format [[prob_class0, prob_class1]]
        probs_list = probabilities.cpu().squeeze().tolist() 
    else:
        # Fallback if it's not a tensor (though it should be from model output)
        probs_list = probabilities 

    # Format probabilities as percentages, e.g., "Não Vulnerável: 0.06%, Vulnerável: 99.94%"
    if len(probs_list) == 2:
        prob_non_vulnerable = probs_list[0] * 100
        prob_vulnerable = probs_list[1] * 100
        formatted_probs = (f"Não Vulnerável: {prob_non_vulnerable:.2f}%, "
                           f"Vulnerável: {prob_vulnerable:.2f}%")
    else:
        # Handle cases where the format might be unexpected
        formatted_probs = f"Formato de probabilidades inesperado: {probs_list}"

    print(f"  Probabilidades: {formatted_probs}")
