import torch

def get_llm_activations(model, tokenizer, texts, layer_index=-1):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model.eval()
    model.to(device)
    all_activations = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            layer_activations = hidden_states[layer_index + 1] 
            
            cls_activations = layer_activations[:, 0, :]
            all_activations.append(cls_activations)
    return torch.cat(all_activations, dim=0)