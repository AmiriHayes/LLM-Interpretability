import requests
import json
import pandas as pd
from pathlib import Path
from nltk.tokenize import sent_tokenize

def retrieve_all_sentences():
    current_dir = Path(__file__).parent.resolve()
    data_dir = current_dir.parent / "data"

    story_sentences = []
    math_sentences = []
    generic_sentences = []
    qa_sentences = []
    multilingual_sentences = []

    # STORY SENTENCES
    story_file = data_dir / "small_text.csv"
    if story_file.exists():
        df = pd.read_csv(story_file)
        sentences = []
        for paragraph in df['text']:
            sentences.extend(sent_tokenize(paragraph))
        story_sentences = sentences[:10_000]
    else:
        print(f"Warning: Could not find {story_file}")

    # GENERIC SENTENCES
    generic_file = data_dir / "generic_sentences.json"
    if generic_file.exists():
        df_json = pd.read_json(generic_file)
        generic_sentences = df_json[0].tolist()

    # MATH SENTENCES
    math_url = "https://datasets-server.huggingface.co/rows?dataset=HuggingFaceH4%2FMATH-500&config=default&split=test&offset=0&length=100"
    response = requests.get(math_url)
    
    if response.status_code == 200:
        data_json = response.json()
        for i, item in enumerate(data_json['rows']):
            math_sentences.append(item['row']['problem'])

    # QA SENTENCES
    # coming soon

    # MULTILINGUAL SENTENCES
    # coming soon

    # CODE SENTENCES
    # coming soon

    return [story_sentences, math_sentences, generic_sentences, qa_sentences, multilingual_sentences]