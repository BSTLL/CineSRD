import os
import json
import yaml
import pandas as pd

def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_prompt(prompt_path="prompt.txt"):
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read()

def instruct(number, drama, config, prompt_template):
    data_root = config['data_root']
    slice_subdir = config['slice_subdir']
    csv_subdir = config['csv_subdir']
    prompt_subdir = config['prompt_subdir']
    seg_size = config['seg_size']
    overlap = config['overlap']
    audio_tag = config['audio_tag']

    csv_path = os.path.join(data_root, csv_subdir, drama, f"{number}.csv")
    outputpath = os.path.join(data_root, prompt_subdir, drama)
    prefix = os.path.join(data_root, slice_subdir, drama, number)
    df = pd.read_csv(csv_path)
    texts = df['text'].tolist()
    audios = df['files'].tolist()
    # 生成音频绝对路径
    new_audios = [os.path.join(prefix, os.path.basename(a)) for a in audios]
    n = len(texts)
    res = []
    start = 0
    while start < n:
        end = min(start + seg_size, n)
        segment = texts[start:end]
        aud = new_audios[start:end]
        block = '\n'.join(f"{s}{audio_tag}" for s in segment)
        curr_prompt = prompt_template.replace('{x}', block)
        temp = {
            'query': curr_prompt,
            'audios': aud,
            'response': ""
        }
        res.append(temp)
        if end == n:
            break
        start = start + seg_size - overlap
    if not os.path.exists(outputpath):
        os.makedirs(outputpath, exist_ok=True)
    file_name = os.path.join(outputpath, f"{number}.json")
    with open(file_name, 'w', encoding='utf-8') as output_file:
        json.dump(res, output_file, ensure_ascii=False, indent=2)

def main():
    config = load_config("config.yaml")
    prompt_template = load_prompt(config.get("prompt_path", "prompt.txt"))
    numbers = config.get('numbers', ['01', '02'])
    dramas = config['dramas']
    for number in numbers:
        print(f"Processing {number}")
        for drama in dramas:
            instruct(number, drama, config, prompt_template)

if __name__ == "__main__":
    main()
