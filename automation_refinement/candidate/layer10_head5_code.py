from transformers import PreTrainedTokenizerBase
import numpy as np

def dominant_theme_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    word_attention_scores = {} # Initial empty dictionary

    # Custom logic inspired by inspecting words associated with top attention in examples.
    # In our hypothesis we posit that this function ranks specific words that dominate the general theme of a sentence.
    for example in examples:
        attention_scores = example["sentence attention"].split(', ')
        for score in attention_scores:
            word, value = score.split('|')[0], float(score.split('|')[-1])
            if word in word_attention_scores:
                word_attention_scores[word] += value
            else:
                word_attention_scores[word] = value

    # For simplicity, dominant theme words are placeholders for high attention scores. In practice,
    # they are evaluated dynamically per sentence fed in. Here we generalize with simplistic logic.
    dominant_words = [word for word, score in word_attention_scores.items() if score > 60]

    # Iterate over tokens to encode attention pattern
    for i, token_id in enumerate(toks.input_ids[0]):
        token_str = tokenizer.convert_ids_to_tokens(token_id.item())
        if token_str in dominant_words:
            out[i, i] = 0.9
        else:
            out[i, i] = 0.1

    # Normalize and include special token self-attention
    out[0, 0] = 1
    out[-1, 0] = 1
    out /= out.sum(axis=1, keepdims=True)

    return "Dominant Theme Attention Pattern", out