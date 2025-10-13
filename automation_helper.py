# helper functions for automating head analysis

import requests
import numpy as np
from typing import Callable, Tuple, Optional
import torch
import json
import importlib.util
import types
from transformers import AutoModel, AutoTokenizer

####### FUNCTION #1: GENERATE AUTOMATED PROMPTS FROM HEAD DATA #######

example_one = """
def dependencies(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]: /
    toks = tokenizer([sentence], return_tensors="pt") /
    len_seq = len(toks.input_ids[0]) /
    out = np.zeros((len_seq, len_seq)) /
    words = sentence.split() /
    doc = nlp(" ".join(words)) /
    for stok in doc: /
        parent_index = stok.i /
        for child_stok in stok.children: /
            child_index = child_stok.i /
            out[parent_index+1, child_index+1] = 1 /
            out[child_index+1, parent_index+1] = 1 /
    for row in range(len_seq): # Ensure no row is all zeros /
        if out[row].sum() == 0: /
            out[row, -1] = 1.0 /
    out += 1e-4  # Avoid division by zero /
    out = out / out.sum(axis=1, keepdims=True)  # Normalize
    return "Dependency Parsing Pattern", out /
"""
example_two = """
def same_attention(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]: /
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    for i in range(1, len_seq-1):
        out[i, i] = 1
    for row in range(len_seq): # Ensure no row is all zeros /
        if out[row].sum() == 0: /
            out[row, -1] = 1.0 /
    return "Same Token Pattern", out
"""
example_three = """
def pos_alignment(sentence: str, tokenizer: PreTrainedTokenizerBase) -> Tuple[str, np.ndarray]:
    toks = tokenizer([sentence], return_tensors="pt") /
    len_seq = len(toks.input_ids[0]) /
    out = np.zeros((len_seq, len_seq)) /
    # assign toks, input_ids, word_ids, len_seq, out, doc /
    # use spacey to get pos_tags for tokens in docs [token.pos_ for token in doc] /
    # for token in pos_tags: /
    # loop through pos_tags and increment out[i,j] when pos_tags match /
    # assign cls (out[0, 0] = 1) and eos (out[-1, 0] = 1) to have self_attention /
    # Normalize out matrix by row (results in uniform attention) and return out /
    for row in range(len_seq): # Ensure no row is all zeros /
        if out[row].sum() == 0: /
            out[row, -1] = 1.0 /
    # return 'Part of Speech Implementation 1', out /
"""

def generate_prompt(sentences, model, tokenizer, head_loc, top_k_ratio=0.1):
    layer, head = head_loc
    data = {
        "layer": layer,
        "head": head,
        "model": model.config.architectures[0],
        "examples": []
    }

    def handle_score(score):
        return "{:.0f}".format(score * 100)
        
    def scrape_head(att, tokens, top_k_ratio, ignore_special=True):
        seq_len = att.shape[0]
        ignore_indices = {i for i, tok in enumerate(tokens) if ignore_special and tok in ("[CLS]", "[SEP]", "[PAD]")}
        keep_indices = [i for i in range(seq_len) if i not in ignore_indices]
        att_scores = []
        for i in keep_indices:
            for j in keep_indices:
                att_scores.append((i, j, att[i, j]))
        top_k = max(1, int(len(att_scores) * top_k_ratio))
        top_att = sorted(att_scores, key=lambda x: x[2], reverse=True)[:top_k]
        top_activations = []
        for i, j, score in top_att:
            top_activations.append(f"[{str(tokens[i])}|{str(tokens[j])}:{handle_score(score)}]")
        top_activations_str = " ".join(top_activations).replace("[", "").replace("]", "")
        return top_activations_str
    
    for idx, sentence in enumerate(sentences):
        inputs = tokenizer(sentence, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            att = outputs.attentions[layer][0, head]
        att = att.detach().cpu().numpy()
        top_activations = scrape_head(att, tokens, top_k_ratio=top_k_ratio)
        item = {f"sentence {idx}": " ".join(tokens), "sentence attention": top_activations}
        data["examples"].append(item)

    data = json.dumps(data, indent=2)
    prompt = f"""
    Using the following pieces of data based on {len(sentences)} sentences, generate three hypothesises about the linguistic role the following head is responsible for based on patterns
    in the activations.  Then, choose the most fitting hypothesis for the head function using examples from the data. Finally, using the linguistic hypothesis you determine, 
    write a python function which takes in a sentence and tokenizer as parameters and outputs the name of the pattern you hypothesize along with a predicted_matrix (size: token_len * token_len), which is the 
    rule encoded matrix mirroring attention patterns you'd predict for any given sentence for Layer {layer}, Head {head}. Feel free to encode complex functions but write the simplest algorithm that captures your 
    observed pattern. You must respond to this prompt in JSON in the form "{{"hypothesis": "...", "program": "..."}} with your chosen hypothesis. Think carefully before generating any code.
    The first portion of your response has key "hypothesis" with the title of the hypothesis and the second portion of your response with key "program" should have valid python code starting with ```python and including imports. These patterns can be simple or 
    complex.  For uniformity, the first three lines of your function should be 'toks = tokenizer([sentence], return_tensors="pt") len_seq = len(toks.input_ids[0]) out = np.zeros((len_seq, len_seq))'.
    Make sure the token sequences from your tokenizer and spaCy (if you must use spaCy) are aligned via a dictionary if necessary, because they split text differently. Make sure you generalize your hypothesis pattern to any sentence. Functions can almost 
    always be expressed in fewer than 50 lines of code. As examples, it has been discovered one head is responsible for the complex task of dependency parsing. It's simplistic predicted pseudocode looks like: 
    {example_one}. Example 2: '''{example_two}''' Example 3: '''{example_three}'''. DATA: {data}"""
    return ' '.join(prompt.strip().split())


####### FUNCTION #2:GENERATE HYPOTHESIS, EXPLANATION, AND PROGRAM SYNTHESIS CODE #######

def parse_llm_idea(prompt, config="YOUR_API_CONFIG", verbalize=True):
    def make_request():
        headers = config["headers_fn"](config["key"])
        payload = config["payload_fn"](prompt)
        response = requests.post(config["url"], headers=headers, data=json.dumps(payload))
        response.raise_for_status()

        if config["model"] == "gemini":
            data = response.json()
            output = data["candidates"][0]["content"]["parts"][0]["text"]
        if config["model"] == "openai":
            pass
        if config["model"] == "claude":
            data = response.json()
            output = data["content"]["text"]
        if config["model"] == "deepseek":
            pass

        return output
    
    output = make_request()

    try:
        result = json.loads(output)

        if type(result) is list: result = result[0]
        hypothesis = result.get("hypothesis", "")
        program = result.get("program", "")

        if program.startswith("```python"): program = program[9:]
        if program.endswith("```"): program = program[:-3]
        program = program.strip()

        if verbalize: print("Hypothesis, Explanation & Program successfully parsed")

    except Exception as e:
        print(f"Parsing API failed: {str(e)}")
        return str(e)

    return hypothesis, program

####### FUNCTION #3: CHECK WHETHER PROGRAM SYNTHESIS IS VALID #######

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m)))

def score_prediction(torch_model, torch_tokenizer, head_loc, pattern, sentence_1, distance="jsd"):
    layer, head = head_loc
    tokens = torch_tokenizer(sentence_1, return_tensors="pt")

    att = torch_model(**tokens, output_attentions=True).attentions[layer][0, head].detach().numpy()
    _, pred_att = pattern(sentence_1, torch_tokenizer)

    if distance == "raw":
        score = np.abs(att - pred_att).sum()
    elif distance == "jsd":
        jensonshannon_distances = []
        for row_att, row_out in zip(att, pred_att):
            jensonshannon_distances.append(np.sqrt(js_divergence(row_att, row_out)))
        score = np.mean(jensonshannon_distances)
    return score

import traceback
def validate_program(program_path, model, tokenizer, layer, head, sentence):       
    try:
        spec = importlib.util.spec_from_file_location("loaded_program", program_path)
        module = importlib.util.module_from_spec(spec)
        module.__dict__['np'] = np
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Program loading failed: {str(e)}")
        return str(e)

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, types.FunctionType):
            program = attr
            break

    try:
        score = score_prediction(model, tokenizer, (layer, head), program, sentence, distance="jsd", output=False)
        return score
    except Exception as e:
        print("hello")
        print(f"Program validation failed: {str(e)}")
        print(traceback.format_exc())
        return str(e)