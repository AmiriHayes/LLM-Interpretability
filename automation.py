from automation_helper import generate_prompt, parse_llm_idea, validate_program
from transformers import AutoModel, AutoTokenizer
import os
import pandas as pd
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize

model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

load_dotenv()
API_CONFIGS = {
    "gemini": {
        "model": "gemini",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        "key": os.getenv("GEMINI"),
        "headers_fn": lambda key: {"Content-Type": "application/json", "X-goog-api-key": key},
        "payload_fn": lambda prompt: {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json"}
        },
        "usage": "https://aistudio.google.com/apikey"
    },
    "openai": {
        "model": "openai",
        "url": "https://api.openai.com/v1/responses",
        "key": os.getenv("OPENAI"),
        "headers_fn": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload_fn": lambda prompt: {"model": "gpt-4.1", "input": prompt},
        "usage": "https://platform.openai.com/account/api-keys"
    },
    "claude": {
        "model": "claude",
        "url": "https://api.anthropic.com/v1/messages",
        "key": os.getenv("CLAUDE"),
        "headers_fn": lambda key: {"x-api-key": key, "Content-Type": "application/json", "Anthropic-Version":"2023-06-01"},
        "payload_fn": lambda prompt: {"model":"claude-sonnet-4-20250514", "messages":[{"role":"user","content":prompt}]},
        "usage": "https://platform.claude.com/api_keys"
    },
    "deepseek": {
        "model": "deepseek",
        "url": "https://api.deepseek.com/chat/completions",
        "key": os.getenv("DEEPSEEK"),
        "headers_fn": lambda key: {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        "payload_fn": lambda prompt: {"model": "deepseek-chat", "input": prompt, "max_tokens": 1000},
        "usage": "https://platform.deepseek.com/api_keys"
    }
}
config = API_CONFIGS["gemini"] 

df_json = pd.read_json('data/generic_sentences.json')
sentences = df_json[0].tolist()

prompts = {}
programs = {}
scores = {}
max_attempts = 1
failed_heads = []

save_folder = "automation_results_gemini"
if not os.path.exists(save_folder): 
    os.makedirs(save_folder)

heads = model.config.num_attention_heads
layers = model.config.num_hidden_layers
for (layer, head) in zip(range(layers), range(heads)):
    # if (layer, head) != (3, 9): continue

    attempts = 0

    prompt_path = f"{save_folder}/prompts/{layer}/"
    program_path = f"{save_folder}/llm_code/programs-layer_{layer}/"
    score_path = f"{save_folder}/scores/"
    os.makedirs(prompt_path, exist_ok=True)
    os.makedirs(program_path, exist_ok=True)
    os.makedirs(score_path, exist_ok=True)

    prompt = generate_prompt(sentences, model, tokenizer, (layer, head), top_k_ratio=0.1)
    with open(f"{prompt_path}/{layer}_{head}_prompt.txt", "w") as f: f.write(prompt)

    config = API_CONFIGS["gemini"] 
    hypothesis, program = parse_llm_idea(prompt, config=config, verbalize=False)

    python_path = f"{program_path}/{head}_output.py"
    with open(python_path, "w") as f: f.write(program)
    feedback = validate_program(python_path, model, tokenizer, layer, head, sentences)
    with open(f"{score_path}/{layer}_{head}_score.txt", "w") as f: f.write(str(feedback))

    while type(feedback) is str or feedback < 0.6:
        if feedback == "error": prompt += f"The program failed to execute with feedback {feedback}. Try again."
        else: program += f"The program had similarity score {feedback} using Jensen Shannon Distance. Try to generate something better."
        hypothesis, program = parse_llm_idea(prompt, config=config, verbalize=False)
        feedback = validate_program(program, model, tokenizer, layer, head, sentences)

        attempts += 1
        if attempts >= max_attempts: 
             failed_heads.append((layer, head, feedback))
             break
    
    prompts[(layer, head)] = prompt
    programs[(layer, head)] = program
    scores[(layer, head)] = feedback

    print(f"Layer {layer}, Head {head} | Score: {feedback} | Hypothesis ~ {hypothesis} ")