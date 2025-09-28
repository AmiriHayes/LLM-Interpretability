import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")
from typing import Optional, Tuple, Callable
from transformers import PreTrainedTokenizerBase
from sklearn.linear_model import LinearRegression

# Docs Link: { https://amirihayes.github.io/LLM-Interpretability/ }
# Unless otherwise notes, these functions are  manually written:

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

#13/30
def verb_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0].tolist()
    len_seq = len(input_ids)
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)
    words = tokenizer.convert_ids_to_tokens(input_ids)
    for i, token in enumerate(words):
        #check if the token is a verb
        if doc[i].pos_ == "VERB":
            print(f"Found verb: {token} at index {i}")
        # if token in verbs:
        #     out[i, i] = 1
            # out[i, i+1] = 0.5
            # out[i, i-1] = 0.5
            # out[i-1, i] = 0.5
            # out[i+1, i] = 0.5
    out[0, 0] = 1
    out[-1, 0] = 1
    out = out / out.sum(axis=1, keepdims=True)
    return "Verb Clustering Pattern", out

#14/30
def noun_modifier_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0].tolist()
    len_seq = len(input_ids)
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ == "NOUN":
            noun_indices = [i for i, tok in enumerate(input_ids) if tok == token.i + tokenizer.vocab_size]
            for i in noun_indices:
                for child in token.children:
                    if child.pos_ in ["ADJ", "DET", "NUM"]:
                        child_indices = [j for j, tok in enumerate(input_ids) if tok == child.i + tokenizer.vocab_size]
                        for j in child_indices:
                            out[i, j] = 1 / len(noun_indices)
    out[0, 0] = 1
    out[-1, 0] = 1
    return "Noun Clustering Pattern", out

#15/30
def pronoun_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0].tolist()
    len_seq = len(input_ids)
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ == "PRON":
            pronoun_indices = [i for i, tok in enumerate(input_ids) if tok == token.i + tokenizer.vocab_size]
            for i in pronoun_indices:
                out[i, :] = 1 / len_seq
    out[0, 0] = 1
    out[-1, 0] = 1
    return "Pronoun Clustering Pattern", out

#16/30
def preposition_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0].tolist()
    len_seq = len(input_ids)
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ == "ADP":
            preposition_indices = [i for i, tok in enumerate(input_ids) if tok == token.i + tokenizer.vocab_size]
            for i in preposition_indices:
                out[i, :] = 1 / len_seq
    out[0, 0] = 1
    out[-1, 0] = 1
    return "Preposition Clustering Pattern", out

#17/30
def adjective_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0].tolist()
    len_seq = len(input_ids)
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ == "ADJ":
            adjective_indices = [i for i, tok in enumerate(input_ids) if tok == token.i + tokenizer.vocab_size]
            for i in adjective_indices:
                out[i, :] = 1 / len_seq
    out[0, 0] = 1
    out[-1, 0] = 1
    return "Adjective Clustering Pattern", out

#18/30
def chainofthought_pattern(sentence: str, tokenizer: PreTrainedTokenizerBase, att: np.ndarray, hint: bool) -> Tuple[str, str, np.ndarray]:
    out = []
    output = False

    prefix = "system\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nuser\n\n"
    if hint == False:
        i = sentence.find("assistant")
        prompt = sentence[:i].strip()
        prompt = prompt[len(prefix):]
    elif hint == True:
        i = sentence.find(" [ Note:")
        prompt = sentence[:i].strip()
        prompt = prompt[len(prefix):]
    len_toks = len(tokenizer([prompt], return_tensors="pt").input_ids[0])
    start_token_idx = len(tokenizer([prefix], return_tensors="pt").input_ids[0])
    prompt_matrix = att[start_token_idx:len_toks, start_token_idx:len_toks]
    vector_1 = np.mean(prompt_matrix, axis=0)
    if output: print(f"Prompt shape: {prompt_matrix.shape}, vector shape: {vector_1.shape}")

    answer = str(sentence.split(".")[-2]).strip()
    toks = tokenizer([sentence], return_tensors="pt")
    decoded_tokens = tokenizer.convert_ids_to_tokens(toks.input_ids[0].tolist())
    period_indices = [i for i, token in enumerate(decoded_tokens) if '.' in token]
    start_idx = 0
    len_toks = len(tokenizer([sentence], return_tensors="pt").input_ids[0])
    if len(period_indices) >= 2:
        second_to_last_period_idx = period_indices[-2]
        start_idx = second_to_last_period_idx
    answer_matrix = att[start_idx:len_toks, start_idx:len_toks]
    vector_2 = np.mean(answer_matrix, axis=0)
    max_token = 20
    if len(vector_2) > max_token:
        vector_2 = vector_2[-max_token:]
    elif len(vector_2) < max_token:
        padding_length = max_token - len(vector_2)
        vector_2 = np.pad(vector_2, (0, padding_length), 'constant', constant_values=0)
    if output: print(f"Answer shape: {answer_matrix.shape}, vector shape: {vector_2.shape}")

    if output: print(f"prompt: {prompt}\nanswer: {answer}\n")
    out = np.concatenate((vector_1, vector_2))
    return prompt, answer, out

#19/30
def negation_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0].tolist()
    len_seq = len(input_ids)
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)
    for token in doc:
        if token.dep_ == "neg":
            token_idx = sentence.split().index(token.text)
            out[token_idx, :] = 1 
    out[0, 0] = 1
    out[-1, 0] = 1
    return "Negation Token Pattern", out

#20/30
def coreference_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0].tolist()
    len_seq = len(input_ids)
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)
    coref_clusters = doc._.coref_clusters if doc._.has_coref else []
    for cluster in coref_clusters:
        representative = cluster.main.text
        representative_indices = [i for i, tok in enumerate(input_ids) if tok == representative]
        for i in representative_indices:
            out[i, :] = 1 / len_seq
            for mention in cluster.mentions:
                mention_indices = [j for j, tok in enumerate(input_ids) if tok == mention.text]
                for j in mention_indices:
                    out[i, j] = 1 / len_seq
                    out[j, i] = 1 / len_seq
    out[0, 0] = 1
    out[-1, 0] = 1
    return "Coreference Token Pattern", out

#21/30
def constituent_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0].tolist()
    len_seq = len(input_ids)
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)
    for token in doc:
        if token.dep_ in ["NP", "VP", "PP", "ADJP", "ADVP"]:
            constituent_indices = [i for i, tok in enumerate(input_ids) if tok == token.i + tokenizer.vocab_size]
            for i in constituent_indices:
                out[i, :] = 1 / len_seq
    out[0, 0] = 1
    out[-1, 0] = 1
    return "Constituent Token Pattern", out

#22/30
def single_token_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0].tolist()
    len_seq = len(input_ids)
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)
    root_index = next(i for i, token in enumerate(doc) if token.dep_ == "ROOT")
    out[:, root_index] = 1
    return "Single Token Pattern", out

#23/30
def root_cluster_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0].tolist()
    len_seq = len(input_ids)
    out = np.zeros((len_seq, len_seq))
    doc = nlp(sentence)
    root_index = next(i for i, token in enumerate(doc) if token.dep_ == "ROOT")
    print(sentence.split(" ")[root_index])
    for i in range(len_seq):
        if i == root_index:
            out[i, i] = 1
            out[i, i-1] = 1
            out[i-1, i] = 1
            out[i, i + 1] = 1
            out[i + 1, i] = 1
        else:
            out[i, -1] = 1
    out = out / out.sum(axis=1, keepdims=True)
    return "Root Cluster Attention Pattern", out


# INITIAL AUTOMATED / LLM-GENERATED FILTERS

def direct_object_prepositional_object_alignment(sentence, tokenizer):
    """
    Hypothesizes that Layer 7, Head 1 is responsible for aligning verbs and prepositions
    with their direct or prepositional objects.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object (e.g., AutoTokenizer.from_pretrained("bert-base-uncased")).

    Returns:
        tuple: A string describing the pattern and a 2D numpy array
               representing the predicted attention matrix.
    """
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0]
    token_len = len(input_ids)
    predicted_matrix = np.zeros((token_len, token_len))

    # Get word IDs to align with spaCy tokens
    word_ids = toks.word_ids()

    # Process sentence with spaCy
    doc = nlp(sentence)

    # Map spaCy token indices to BERT token indices
    spacy_to_bert_map = {}
    bert_to_spacy_map = {}
    current_spacy_token_idx = -1
    for bert_idx, word_id in enumerate(word_ids):
        if word_id is not None and (current_spacy_token_idx == -1 or word_id != word_ids[bert_idx - 1]):
            current_spacy_token_idx = word_id
            spacy_to_bert_map[current_spacy_token_idx] = bert_idx
            bert_to_spacy_map[bert_idx] = current_spacy_token_idx
        elif word_id is not None:
            bert_to_spacy_map[bert_idx] = current_spacy_token_idx

    # Iterate through spaCy tokens to find verbs and prepositions and their objects
    for i, token in enumerate(doc):
        # Find BERT index for the current spaCy token
        from_bert_idx_start = -1
        for bert_idx, spacy_id in bert_to_spacy_map.items():
            if spacy_id == i:
                from_bert_idx_start = bert_idx
                break

        if from_bert_idx_start == -1: # Skip if spaCy token doesn't map to BERT token
            continue

        # Look for direct objects (dobj) or prepositional objects (pobj)
        if token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ == "dobj":
                    # Distribute attention from the verb to its direct object tokens
                    to_bert_idx_start = -1
                    for bert_idx, spacy_id in bert_to_spacy_map.items():
                        if spacy_id == child.i:
                            to_bert_idx_start = bert_idx
                            break
                    if to_bert_idx_start != -1:
                        # Find all BERT tokens that correspond to the spaCy child token
                        bert_indices_for_child = [b_idx for b_idx, s_id in bert_to_spacy_map.items() if s_id == child.i]
                        if bert_indices_for_child:
                            # Assign high attention from the 'from' BERT token (verb)
                            # to all BERT tokens that form the 'to' (object)
                            for to_b_idx in bert_indices_for_child:
                                predicted_matrix[from_bert_idx_start, to_b_idx] = 0.8 # High weight

        elif token.pos_ == "ADP":  # Adposition (preposition or postposition)
            for child in token.children:
                if child.dep_ == "pobj":
                    # Distribute attention from the preposition to its object tokens
                    to_bert_idx_start = -1
                    for bert_idx, spacy_id in bert_to_spacy_map.items():
                        if spacy_id == child.i:
                            to_bert_idx_start = bert_idx
                            break
                    if to_bert_idx_start != -1:
                        bert_indices_for_child = [b_idx for b_idx, s_id in bert_to_spacy_map.items() if s_id == child.i]
                        if bert_indices_for_child:
                            for to_b_idx in bert_indices_for_child:
                                predicted_matrix[from_bert_idx_start, to_b_idx] = 0.8 # High weight

    # Add self-attention for [CLS] and [SEP] tokens
    predicted_matrix[0, 0] = 1.0
    predicted_matrix[token_len - 1, token_len - 1] = 1.0

    # For any row where no attention has been assigned, distribute attention uniformly
    # or assign to [CLS] for general context
    for i in range(token_len):
        if np.sum(predicted_matrix[i, :]) == 0:
            # Fallback: if no specific object found, distribute attention somewhat broadly
            # or assign to CLS for general context (this is a heuristic)
            predicted_matrix[i, 0] = 0.5 # Attend to CLS for general context
            predicted_matrix[i, i] = 0.5 # Self-attention

    # Normalize each row to sum to 1
    for i in range(token_len):
        row_sum = np.sum(predicted_matrix[i, :])
        if row_sum > 0:
            predicted_matrix[i, :] = predicted_matrix[i, :] / row_sum

    return 'Direct Object / Prepositional Object Alignment', predicted_matrix

def determiner_noun_phrase_linking(sentence: str, tokenizer) -> tuple[str, np.ndarray]:
    """
    Hypothesizes attention patterns where determiners link to the nouns
    and adjectives within their associated noun phrases.

    Args:
        sentence (str): The input sentence.
        tokenizer: The tokenizer object (e.g., from Hugging Face Transformers).

    Returns:
        tuple[str, np.ndarray]: A tuple containing the name of the pattern
                                and the predicted attention matrix.
    """
    toks = tokenizer([sentence], return_tensors="pt")
    input_ids = toks.input_ids[0]
    token_len = len(input_ids)
    predicted_matrix = np.zeros((token_len, token_len))

    # Get spaCy doc for linguistic analysis
    doc = nlp(sentence)

    # Create a mapping from tokenizer's token indices to spaCy's token indices
    # This is crucial for aligning the attention matrix with linguistic features.
    # The tokenizer's `word_ids` method is ideal for this.
    word_ids = toks.word_ids(batch_index=0) # Get word_ids for the first (and only) sentence in the batch

    for i in range(token_len):
        current_word_idx = word_ids[i]
        if current_word_idx is not None and current_word_idx < len(doc):
            spacy_token = doc[current_word_idx]

            # If the current token (from the tokenizer) corresponds to a determiner in spaCy
            if spacy_token.pos_ == "DET":
                # Find the head of the determiner (typically the noun it modifies)
                head_spacy_token = spacy_token.head

                # Attend from the determiner's subword token(s) to its head's subword token(s)
                for j in range(token_len):
                    target_word_idx = word_ids[j]
                    if target_word_idx is not None and target_word_idx == head_spacy_token.i:
                        predicted_matrix[i, j] = 1.0

                # Also attend from the determiner's subword token(s) to any adjectives
                # that are children of the head and appear before the head
                for child in head_spacy_token.children:
                    if child.pos_ == "ADJ" and child.i < head_spacy_token.i:
                        for j in range(token_len):
                            target_word_idx = word_ids[j]
                            if target_word_idx is not None and target_word_idx == child.i:
                                predicted_matrix[i, j] = 1.0


    # Apply self-attention for [CLS] and [SEP] tokens
    predicted_matrix[0, 0] = 1.0
    predicted_matrix[token_len - 1, token_len - 1] = 1.0

    # Normalize rows to sum to 1 to represent attention probabilities
    # Avoid division by zero for rows that might still be all zeros (e.g., padding tokens)
    row_sums = predicted_matrix.sum(axis=1, keepdims=True)
    predicted_matrix = np.where(row_sums == 0, 0, predicted_matrix / row_sums)

    return "Determiner-Noun/Adjective-Noun Phrase Linking", predicted_matrix

def verb_phrase_modifier_attention(sentence: str, tokenizer) -> tuple[str, np.ndarray]:
    """
    Hypothesizes the attention pattern for a head responsible for connecting
    verbs to their related phrases and modifiers (subjects, objects, adverbs, PPs).

    Args:
        sentence: The input sentence.
        tokenizer: The tokenizer object (e.g., from Hugging Face Transformers).

    Returns:
        A tuple containing:
            - The name of the hypothesized pattern.
            - A NumPy array (predicted_matrix) representing the rule-encoded
              attention pattern.
    """
    # Load the English NLP model for spaCy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading en_core_web_sm model for spaCy. Please run 'python -m spacy download en_core_web_sm' once.")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # Tokenize the sentence using the provided tokenizer
    tokens = tokenizer([sentence], return_tensors="pt")
    input_ids = tokens.input_ids[0].tolist()
    token_ids = tokenizer.convert_ids_to_tokens(input_ids)

    len_seq = len(token_ids)
    predicted_matrix = np.zeros((len_seq, len_seq))

    doc = nlp(sentence)

    # Create a mapping from spaCy token index to BERT token indices
    # This handles WordPiece tokenization where one spaCy token might be multiple BERT tokens
    spacy_to_bert_map = []
    current_bert_idx = 1  # Start after [CLS]

    for spacy_token in doc:
        # Tokenize the spaCy token to get its BERT sub-tokens
        bert_sub_tokens = tokenizer.tokenize(spacy_token.text)
        bert_indices_for_spacy_token = list(range(current_bert_idx, current_bert_idx + len(bert_sub_tokens)))
        spacy_to_bert_map.append(bert_indices_for_spacy_token)
        current_bert_idx += len(bert_sub_tokens)

    # Iterate through spaCy tokens to identify verbs and their relations
    for i, spacy_token in enumerate(doc):
        # Get the BERT indices corresponding to the current spaCy token
        from_bert_indices = spacy_to_bert_map[i]

        # Prioritize attention to verb and its direct dependents
        if spacy_token.pos_ == "VERB":
            # Direct attention from the verb to its subject (nsubj) and direct object (dobj)
            for child in spacy_token.children:
                if child.dep_ in ["nsubj", "dobj", "iobj", "attr", "acomp", "xcomp", "prep", "advcl", "advmod"]:
                    if child.i < len(spacy_to_bert_map): # Ensure child index is within bounds
                        to_bert_indices = spacy_to_bert_map[child.i]
                        for from_idx in from_bert_indices:
                            for to_idx in to_bert_indices:
                                if from_idx < len_seq and to_idx < len_seq:
                                    predicted_matrix[from_idx, to_idx] = 1.0

            # Also attend from the verb to itself for self-attention
            for idx in from_bert_indices:
                if idx < len_seq:
                    predicted_matrix[idx, idx] = 1.0

        # Prioritize attention from subjects/adverbs/prepositions to their governing verb
        elif spacy_token.dep_ in ["nsubj", "advmod", "prep", "aux", "auxpass"]:
            if spacy_token.head and spacy_token.head.pos_ == "VERB":
                head_bert_indices = spacy_to_bert_map[spacy_token.head.i]
                for from_idx in from_bert_indices:
                    for to_idx in head_bert_indices:
                        if from_idx < len_seq and to_idx < len_seq:
                            predicted_matrix[from_idx, to_idx] = 1.0

        # Prioritize attention from direct objects/complement to their governing verb
        elif spacy_token.dep_ in ["dobj", "iobj", "attr", "acomp", "xcomp", "ccomp", "acl"]:
            if spacy_token.head and spacy_token.head.pos_ == "VERB":
                head_bert_indices = spacy_to_bert_map[spacy_token.head.i]
                for from_idx in from_bert_indices:
                    for to_idx in head_bert_indices:
                        if from_idx < len_seq and to_idx < len_seq:
                            predicted_matrix[from_idx, to_idx] = 1.0

        # Attention from prepositions to the noun phrase they introduce
        elif spacy_token.pos_ == "ADP": # Adposition (preposition or postposition)
            for child in spacy_token.children:
                if child.dep_ == "pobj": # Object of preposition
                    if child.i < len(spacy_to_bert_map):
                        to_bert_indices = spacy_to_bert_map[child.i]
                        for from_idx in from_bert_indices:
                            for to_idx in to_bert_indices:
                                if from_idx < len_seq and to_idx < len_seq:
                                    predicted_matrix[from_idx, to_idx] = 1.0
                # If the preposition is attached to a verb, also attend back to the verb
                if spacy_token.head and spacy_token.head.pos_ == "VERB":
                    head_bert_indices = spacy_to_bert_map[spacy_token.head.i]
                    for from_idx in from_bert_indices:
                        for to_idx in head_bert_indices:
                            if from_idx < len_seq and to_idx < len_seq:
                                predicted_matrix[from_idx, to_idx] = 1.0

        # Adjectives attending to their noun or verb (if copular)
        elif spacy_token.pos_ == "ADJ":
            if spacy_token.head:
                if spacy_token.head.pos_ == "NOUN" or (spacy_token.head.pos_ == "VERB" and spacy_token.dep_ == "acomp"):
                    head_bert_indices = spacy_to_bert_map[spacy_token.head.i]
                    for from_idx in from_bert_indices:
                        for to_idx in head_bert_indices:
                            if from_idx < len_seq and to_idx < len_seq:
                                predicted_matrix[from_idx, to_idx] = 1.0

        # Handle attention from [CLS] and [SEP] tokens
        # [CLS] token (index 0) often has broad attention or self-attention
        predicted_matrix[0, 0] = 1.0
        # [SEP] token (last token) often attends to [CLS] or has self-attention
        if len_seq > 1:
            predicted_matrix[len_seq - 1, 0] = 1.0
            predicted_matrix[len_seq - 1, len_seq - 1] = 1.0

        # Ensure all rows sum to 1 by distributing any remaining attention to [CLS] or [SEP]
    for i in range(len_seq):
        current_row_sum = predicted_matrix[i].sum()
        if current_row_sum == 0:
            # If a row is all zeros, distribute attention to [CLS] and [SEP]
            # or to itself if it's [CLS] or [SEP]
            if i == 0:  # [CLS] token
                predicted_matrix[i, 0] = 1.0
            elif i == len_seq - 1:  # [SEP] token
                predicted_matrix[i, len_seq - 1] = 1.0
            else:
                # For other tokens, distribute attention to [CLS] and [SEP]
                # You could also consider distributing to the token itself or other meaningful global tokens
                predicted_matrix[i, 0] = 0.5
                if len_seq > 1:
                    predicted_matrix[i, len_seq - 1] = 0.5
        else:
            predicted_matrix[i] = predicted_matrix[i] / current_row_sum

    return "Verb-Related Phrase and Modifier Focus", predicted_matrix