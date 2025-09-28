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

example_function_one = """
def dependencies(sentence, tokenizer):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    words = sentence.split()
    # use spacey nlp to split word into doc dependency tree
    # loop through each node in tree and assign directional attention
    # to the matrix 'out' by adding one when there is an outgoing edge.
    # assign cls (out[0, 0] = 1) and eos (out[-1, 0] = 1) to have self_attention
    # Normalize out matrix by row (results in uniform attention) and return out
    return 'Dependency Parsing Pattern', out
"""
example_function_two = """
def pos_alignment(sentence, tokenizer):
    toks = tokenizer([sentence], return_tensors="pt")
    len_seq = len(toks.input_ids[0])
    out = np.zeros((len_seq, len_seq))
    # assign toks, input_ids, word_ids, len_seq, out, doc
    # use spacey to get pos_tags for tokens in docs [token.pos_ for token in doc]
    # for token in pos_tags:
    # loop through pos_tags and increment out[i,j] when pos_tags match
    # assign cls (out[0, 0] = 1) and eos (out[-1, 0] = 1) to have self_attention
    # Normalize out matrix by row (results in uniform attention) and return out
    # return 'Part of Speech Implementation 1', out
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
        return "{:.2f}".format(score)
        
    def scrape_head(att, tokens, ignore_special=True, top_k_ratio=0.1):
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
            top_activations.append({
                f"from_token_{i}": tokens[i],
                f"to_token_{j}": tokens[j],
                "weight": handle_score(score)
            })
        return top_activations
    
    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            att = outputs.attentions[layer][0, head]
        att = att.detach().cpu().numpy()
        top_activations = scrape_head(att, tokens, top_k_ratio=top_k_ratio)
        item = { "sentence": sentence, "attention": top_activations }
        data["examples"].append(item)

    data = json.dumps(data, indent=2)
    prompt = f"""
    Using the following pieces of data based on {len(sentences[0])} sentences, generate three hypothesises about the linguistic role the following head is responsible for based on patterns
    in the activations.  Then, choose the most fitting hypothesis for the head function using a few examples from the data. Finally, using the linguistic hypothesis you determine, 
    write a python function which takes in a sentence and tokenizer as parameters and outputs the name of the pattern you hypothesize along with a 'predicted_matrix' (size: token_len * token_len), which is the 
    rule-encoded matrix mirroring attention patterns you'd predict for any given sentence for Layer {layer}, Head {head}. Feel free to use the capabilities of provided libraries like spacey and nltk for describing 
    linguistic concepts. Feel free to encode complex functions. You must respond to this prompt in JSON in the form "{{"hypothesis": "...", "explanation": "...", "program": "..."}} with your chosen hypothesis.
    The first portion of your response has key "hypothesis" with the title of the hypothesis, the second part has key "explanation" with all explanation text, and the third portion of your response with key "program" 
    should have valid python code. These patterns can be simple or complex. Write the simplest algorithm that captures the pattern. For uniformity, the first three lines of your function should be 'toks = tokenizer([sentence], return_tensors="pt") len_seq = len(toks.input_ids[0]) out = np.zeros((len_seq, len_seq))'.
    Make sure the token sequences from your tokenizer and spaCy (if you must use spaCy) are aligned, because they often split text differently. To avoid tokenization errors, consider looping only up to the length of the shorter sequence and avoid assuming they match. Make sure you generalize your hypothesis pattern to any sentence. As examples: Layer 3, Head 9 has been found to be responsible for dependency parsing. It's predicted pseudocode would look like:
    {example_function_one}. Here is another pseudocode example for one method to implement part-of-speech: {example_function_two}. Make sure you return a valid attention matrix. Here is the data for Layer {layer}, Head {head}: {data}"""
    return ' '.join(prompt.strip().split())


####### FUNCTION #2:GENERATE HYPOTHESIS, EXPLANATION, AND PROGRAM SYNTHESIS CODE #######

def parse_llm_idea(prompt, API_KEY="YOUR_API_KEY", output=True):
    def make_request():
        payload = {
            "contents": [{ "parts": [{"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json"}
        }
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            headers={"Content-Type": "application/json", "X-goog-api-key": API_KEY},
            data=json.dumps(payload)
        )
        return response
    
    if output: print("0: STEP ZERO / THREE: Making LLM Request")
    response = make_request()

    if response.status_code != 200:
        print("Error:", response.status_code, response.text)
    else:
        data = response.json()
        if output: print("1: STEP ONE / THREE: Automated LLM response loaded")
        output_text = data["candidates"][0]["content"]["parts"][0]["text"]
        try:
            result = json.loads(output_text)

            if type(result) is list: result = result[0]
            hypothesis = result.get("hypothesis", "")
            explanation = result.get("explanation", "")
            program = result.get("program", "")
            if program.startswith("```python"): program = program[9:]
            if program.endswith("```"): program = program[:-3]
            program = program.strip()
            if output: print("2: STEP TWO / THREE: Hypothesis, Explanation & Program successfully parsed")

            def validate_program(program):
                # turns out this is cumbersome & mostly unnecessary
                pass

            validate_program(program)
            if output: print("3: STEP THREE / THREE: Attention program validated, process complete")

        except json.JSONDecodeError:
            response = make_request()
            data = response.json()
            if output: print("1: STEP ONE / THREE: Automated LLM response loaded")
            output_text = data["candidates"][0]["content"]["parts"][0]["text"]
            result = json.loads(output_text)

            if type(result) is list: result = result[0]
            hypothesis = result.get("hypothesis", "")
            explanation = result.get("explanation", "")
            program = result.get("program", "")
            if output: print("2: STEP TWO / THREE: Hypothesis, Explanation & Program successfully parsed")

            def validate_program(program):
                pass

            validate_program(program)
            if output: print("3: STEP THREE / THREE: Attention program validated, process complete")

    return hypothesis, explanation, program

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

def validate_program(program_path, model, tokenizer, layer, head, sentences):       
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
        score = score_prediction(model, tokenizer, (layer, head), program, sentences, distance="jsd", output=False)
        return score
    except Exception as e:
        print(f"Program validation failed: {str(e)}")
        return str(e)