import numpy as np
from typing import Optional, Tuple, Callable
from transformers import PreTrainedTokenizerBase
from sklearn.linear_model import LinearRegression
import spacy
nlp = spacy.load("en_core_web_sm")

# Docs Link: { https://amirihayes.github.io/LLM-Interpretability/ }

#0/30
def linear_fit(sentence: str, tokenizer: PreTrainedTokenizerBase, patterns: list[Callable], y: np.ndarray) -> Tuple[str, np.ndarray, float, np.ndarray]:
    X = []
    for pattern in patterns:
      X.append(pattern(sentence, tokenizer)[1].flatten())
    X_n = np.array(X).T
    reg = LinearRegression().fit(X_n, y.flatten())
    out = reg.intercept_ + sum(coef * mat for coef, mat in zip(reg.coef_, X))
    len_seq = len(tokenizer([sentence], return_tensors="pt").input_ids[0])
    out = out.reshape((len_seq, len_seq))
    out = out / out.sum(axis=1, keepdims=True)
    return "Linear Fit Attention", out, reg.intercept_, reg.coef_

# 1/30
def next_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    for i in range(1, len_seq-1):
        out[i, i+1] = 1
    out[0,0] = 1
    out[-1,0] = 1
    return "Next Token Pattern", out

#2/30
def previous_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    for i in range(1, len_seq-1):
        out[i, i-1] = 1
    out[0,0] = 1
    out[-1,0] = 1
    return "Previous Token Pattern", out

#3/30
def same_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    for i in range(1, len_seq-1):
        out[i, i] = 1
    out[0,0] = 1
    out[-1,0] = 1
    return "Same Token Pattern", out

#4/30
def punctuation_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = tokenizer.convert_ids_to_tokens(toks.input_ids[0])
    punctuation_set = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
    punctuation_indices = [i for i, tok in enumerate(words) if any(p in tok for p in punctuation_set)]
    for i in range(len_seq):
        future_punct = [j for j in punctuation_indices if j > i]
        if future_punct:
            for j in future_punct:
                out[i, j] = 1.0
            out[i] /= out[i].sum()
        else:
            out[i, i] = 1.0
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Punctuation Pattern", out

#5/30
def last_token_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    for i in range(len_seq):
        out[i, -1] = 0.5
        out[i, -2] = 0.5
    return "Last Token Pattern", out

#6/30
def repeated_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0].tolist()
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    for i in range(1, len_seq-1):
        token_id = input_ids[i]
        for j in range(1, len_seq-1):
            if input_ids[j] == token_id:
                out[i, j] = 1
    out[0,0] = 1
    out[-1,0] = 1
    out = out / out.sum(axis=1, keepdims=True)
    return "Repitition Pattern", out

#7/30
def uniform_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.ones((len_seq, len_seq)) / len_seq
    return "Uniform Pattern", out

#8/30
def cls_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    out[:, 0] = 1
    return "CLS Pattern", out

#9/30
def eos_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    out[:, -1] = 1
    return "EOS Pattern", out

#10/30
def special_token_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    special_tokens = tokenizer.all_special_ids
    for i in range(len_seq):
        if toks.input_ids[0][i] in special_tokens:
            for sp_tok in special_tokens:
                out[i, toks.input_ids[0] == sp_tok] = 1
        else:
            out[i, -1] = 1
    out = out / out.sum(axis=1, keepdims=True)
    return "Special Token Pattern", out

#11/30
def dependencies(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()
    doc = nlp(" ".join(words))
    check_errors = False
    if check_errors:
        if len(doc) == 0: print("problem, doc empty")
        if len(doc) != (len_seq-2): print("problem, doc length mismatch", len(doc), len(toks)-2)
    for stok in doc:
        parent_index = stok.i
        for child_stok in stok.children:
            child_index = child_stok.i
            out[parent_index+1, child_index+1] = 1
            out[child_index+1, parent_index+1] = 1
    out[0, 0] = 1
    out[-1, 0] = 1
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Dependency Parsing Pattern", out

#12/30
def pos_alignment(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=True)
    input_ids = toks.input_ids[0].tolist()
    word_ids = toks.word_ids(0)
    len_seq = len(input_ids)
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)
    pos_tags = [token.pos_ for token in doc]
    pos_to_token_indices = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx >= len(pos_tags):
            continue
        pos = pos_tags[word_idx]
        pos_to_token_indices.setdefault(pos, []).append(token_idx)
    for token_indices in pos_to_token_indices.values():
        if len(token_indices) > 1:
            for i in token_indices:
                for j in token_indices:
                    out[i, j] = 1
        else:
            i = token_indices[0]
            out[i, i] = 1
    out[0, 0] = 1
    out[-1, -1] = 1
    out += 1e-4
    out = out / out.sum(axis=1, keepdims=True)
    return "Part of Speech Pattern", out