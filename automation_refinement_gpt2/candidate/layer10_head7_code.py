# Import necessary libraries 
import numpy as np 
from transformers import PreTrainedTokenizerBase, GPT2Tokenizer 
import spacy 
# Loading spaCy for linguistic processing 
nlp = spacy.load('en_core_web_sm')

def initial_noun_dominance(sentence: str, tokenizer: PreTrainedTokenizerBase) -> tuple:
    # Tokenize the sentence
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))

    # Use spaCy to parse the sentence
    doc = nlp(sentence)

    # Find the first noun in the sentence
    noun_index = -1
    for tok in doc:
        if tok.pos_ in {"NOUN", "PROPN"}:
            noun_index = tok.i
            break

    # Align spaCy indices with tokenizer indices
    s2t = {}
    i_spacy = 0
    i_toks = 1
    while i_spacy < len(doc) and i_toks < len_seq - 1:
        s2t[i_spacy] = i_toks
        piece = toks.input_ids[0, i_toks].item()
        word_len = len(tokenizer.decode([piece]).strip())
        if len(doc[i_spacy].text) == word_len:
            i_spacy += 1
        i_toks += 1

    if noun_index in s2t:
        first_noun_index = s2t[noun_index]
        # Simulating higher attention values towards the first noun
        for token_idx in range(1, len_seq - 1):
            out[first_noun_index, token_idx] = 1
            out[token_idx, first_noun_index] = 1

    # Set self-attention for CLS and EOS tokens
    out[0, 0] = 1
    out[-1, 0] = 1

    # Normalize the output matrix
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)

    return "Initial Noun Dominance", out

# Example of using the function
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2') 
# sentence = "He packed his bags carefully" 
# result = initial_noun_dominance(sentence, tokenizer) 
# print(result[0]) 
# print(result[1])