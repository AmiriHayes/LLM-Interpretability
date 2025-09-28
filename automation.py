from automation_helper import generate_prompt, parse_llm_idea, validate_program
from transformers import AutoModel, AutoTokenizer
import os
import pandas as pd
from nltk.tokenize import sent_tokenize

API_KEY = "AIzaSyBBW70Rgbm8tCa6l2897Cjq5-jveB9VXhY"
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# df = pd.read_csv('data/small_text.csv')
# sentences = []
# for paragraph in df['text']:
#     sentences.extend(sent_tokenize(paragraph))
# sentences = sentences[:1_000]

df_json = pd.read_json('data/generic_sentences.json')
sentences = df_json[0].tolist()

prompts = {}
programs = {}
scores = {}
max_attempts = 1
failed_heads = []

save_folder = "automation_results"
if not os.path.exists(save_folder): 
    os.makedirs(save_folder)

heads = model.config.num_attention_heads
layers = model.config.num_hidden_layers
for (layer, head) in zip(range(layers), range(heads)):
    if (layer, head) != (3, 9): continue

    attempts = 0

    prompt_path = f"{save_folder}/prompts/{layer}/"
    program_path = f"{save_folder}/llm_code/programs-layer_{layer}/"
    score_path = f"{save_folder}/scores/"
    os.makedirs(prompt_path, exist_ok=True)
    os.makedirs(program_path, exist_ok=True)
    os.makedirs(score_path, exist_ok=True)

    prompt = generate_prompt(sentences, model, tokenizer, (layer, head), top_k_ratio=0.1)
    with open(f"{prompt_path}/{layer}_{head}_prompt.txt", "w") as f: f.write(prompt)
    hypothesis, explanation, program = parse_llm_idea(prompt, API_KEY=API_KEY, output=True)
    python_path = f"{program_path}/{head}_output.py"
    with open(python_path, "w") as f: 
        if program.startswith("```python"): program = program[9:]
        if program.endswith("```"): program = program[:-3]
        program = program.strip()
        f.write(program)
    feedback = validate_program(python_path, model, tokenizer, layer, head, sentences)
    with open(f"{score_path}/{layer}_{head}_score.txt", "w") as f: f.write(str(feedback))

    while type(feedback) is str or feedback < 0.6:
        if feedback == "error": prompt += f"The program failed to execute with feedback {feedback}. Try again."
        else: program += f"The program had similarity score {feedback} using Jensen Shannon Distance. Try to generate something better."
        hypothesis, explanation, program = parse_llm_idea(prompt, API_KEY=API_KEY, output=True)
        feedback = validate_program(program, model, tokenizer, layer, head, sentences)

        attempts += 1
        if attempts >= max_attempts: 
             failed_heads.append((layer, head, feedback))
             break
    
    prompts[(layer, head)] = prompt
    programs[(layer, head)] = program
    scores[(layer, head)] = feedback

    print(f"Layer {layer}, Head {head} | Score: {feedback} | Hypothesis ~ {hypothesis} ")