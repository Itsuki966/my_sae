import torch
from typing import List, Tuple, Dict, Any, Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizer

def get_llm_activations(model: PreTrainedModel, 
                        tokenizer: PreTrainedTokenizer, 
                        texts: List[str], 
                        layer_index: int = -1) -> torch.Tensor:
    """
    テキストのリストからLLMの活性化を抽出する
    
    Args:
        model: Transformersモデル
        tokenizer: モデルに対応するトークナイザー
        texts: 処理するテキストのリスト
        layer_index: 抽出する層のインデックス（負のインデックスは末尾から）
        
    Returns:
        torch.Tensor: [CLS]トークンの活性化を含むテンソル
    """
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

# Residual Streamから活性化を取得
def get_llm_activations_residual_stream(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    texts: List[str], 
    layer_index: int, 
    max_length: int = 256
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    LLMのResidual Stream（残差ストリーム）から活性化を取得
    
    Args:
        model: Transformersモデル 
        tokenizer: モデルに対応するトークナイザー
        texts: 処理するテキストのリスト
        layer_index: 抽出する層のインデックス
        max_length: トークン化する最大長
        
    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
            - 全トークンの活性化を連結したテンソル
            - テキストごとの活性化を格納した辞書
    """
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model.eval()
    model.to(device)
    
    all_token_activations_dict = {}
    all_token_activations = []

    # トークナイザーの準備
    _prepare_tokenizer_for_padding(tokenizer, model)

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, 
                          padding="max_length", return_attention_mask=True)
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
            all_token_activations_dict[text] = actual_token_activations
    
    if not all_token_activations:
        print("No valid token activations found. Returning empty tensor.")
        return torch.empty(0, model.config.hidden_size, device=device), all_token_activations_dict
    
    return torch.cat(all_token_activations, dim=0), all_token_activations_dict

def _prepare_tokenizer_for_padding(tokenizer: PreTrainedTokenizer, model: Optional[PreTrainedModel] = None) -> None:
    """
    トークナイザーにパディングトークンが設定されていない場合に設定する
    
    Args:
        tokenizer: 設定するトークナイザー
        model: トークン埋め込み層のリサイズが必要な場合のモデル
    """
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            if model is not None:
                model.resize_token_embeddings(len(tokenizer))
                print("Added pad token to tokenizer and resized model embeddings.")
            else:
                print("Added pad token to tokenizer.")