def initial_word_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    attention_values = [100, 97, 96, 95, 94, 93, 92, 91]
    for i in range(1, min(len_seq-1, len(attention_values)+1)):
        out[i, 0] = attention_values[i-1] / 100
    out[0, 0] = 1
    out[-1, 0] = 1
    return "Initial Word Attention", out
