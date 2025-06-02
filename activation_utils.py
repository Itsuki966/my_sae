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

def get_llm_activations_residual_stream(model, tokenizer, texts, layer_index, max_length=256):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model.eval()
    model.to(device)
    all_token_activations = []

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else :
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
            print("Added pad token to tokenizer.")

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length", return_attention_mask=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            
        hidden_states = outputs.hidden_states
        layer_activations = hidden_states[layer_index + 1]

        layer_activations_no_batch = layer_activations.squeeze(0)
        attention_mask_no_batch = attention_mask.squeeze(0)

        actual_token_indices = attention_mask_no_batch == 1
        actual_token_activations = layer_activations_no_batch[actual_token_indices]

        if actual_token_activations.shape[0] > 0:
            all_token_activations.append(actual_token_activations)
    if not all_token_activations:
        print("No valid token activations found. Returning empty tensor.")
        return torch.empty(0, layer_activations.shape[-1], device=device)
    
    return torch.cat(all_token_activations, dim=0)