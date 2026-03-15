import os
import json
import yaml
import warnings
import pandas as pd

from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import torch
import librosa

def load_config(config_path="config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_conv(number, drama, prompt_dir):
    data_path = os.path.join(prompt_dir, drama, f"{number}.json")
    with open(data_path, 'r', encoding='utf-8') as f:
        datas = json.load(f)
    conversations = []
    for item in datas:
        query = item['query']
        audio_paths = item['audios']
        content = [{"type": "text", "text": query}]
        for audio_path in audio_paths:
            content.append({"type": "audio", "audio_url": audio_path})
        conversations.append({"role": "user", "content": content})
    return conversations

def get_prob(conversation, processor, model, log_prt=False):
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audios.append(
                        librosa.load(
                            ele['audio_url'],
                            sr=processor.feature_extractor.sampling_rate
                        )[0]
                    )
    from torch.nn.functional import softmax
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
    inputs.input_ids = inputs.input_ids.to(model.device if hasattr(model, 'device') else "cuda")

    output = model.generate(
        **inputs, 
        max_length=256,
        output_scores=True,
        return_dict_in_generate=True,
        do_sample=True,
        top_p=0.95,
        temperature=1.2
    )
    generate_ids = output.sequences[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    scores_list = []
    for i, (token_id, logit) in enumerate(zip(generate_ids[0], output.scores)):
        probs = torch.nn.functional.softmax(logit[0], dim=-1)
        token_str = processor.tokenizer.decode([token_id.item()])
        if i > 0:
            prev_token_id = generate_ids[0][i-1].item()
            prev_token_str = processor.tokenizer.decode([prev_token_id])
            if '<' in prev_token_str:
                token_id_0 = processor.tokenizer.convert_tokens_to_ids('0')
                token_id_1 = processor.tokenizer.convert_tokens_to_ids('1')
                prob_0 = probs[token_id_0].item()
                prob_1 = probs[token_id_1].item()
                s = prob_0 + prob_1
                scores_list.append(float(prob_1) / s if s != 0 else None)
    return scores_list, response

def main_post(drama, config, processor, model):
    seg_size = config['audio_text_infer']['seg_size']
    overlap_half = config['audio_text_infer']['overlap_half']
    csv_dir = config['audio_text_infer']['csv_dir']
    prompt_dir = config['audio_text_infer']['prompt_dir']
    result_dir = config['audio_text_infer']['result_dir']
    for i in range(1, 21):
        number = str(i).zfill(2)
        print(f"Processing {number}")
        csv_file = os.path.join(csv_dir, drama, f"{number}.csv")
        save_path = os.path.join(result_dir, drama)
        os.makedirs(save_path, exist_ok=True)
        conversations = get_conv(number, drama, prompt_dir)
        df = pd.read_csv(csv_file)
        eff_score = [None] * len(df)
        start_idx = 0
        for index, conversation in enumerate(conversations):
            for iter_num in range(5):
                scores, response = get_prob([conversation], processor, model)
                if len(scores) != len(conversation['content'])-1:
                    print(f"[WARN] (iter {iter_num}) index={index}: length not match! scores={len(scores)} content={len(conversation['content'])-1}")
                else:
                    if index == 0:
                        end_idx = start_idx + len(scores) - overlap_half
                        eff_score[start_idx:end_idx] = scores[:-overlap_half]
                    elif index == len(conversations) - 1:
                        eff_score[start_idx:] = scores[overlap_half:]
                    else:
                        end_idx = start_idx + len(scores) - overlap_half * 2
                        eff_score[start_idx:end_idx] = scores[overlap_half:-overlap_half]
                    start_idx = end_idx
                    break
            else:
                print(f"[Error] index={index}: length not match! scores={len(scores)} content={len(conversation['content'])-1}")
                p = conversation['content'][0]
                print(f'conversation: {p}')
                print(f'response:{response}')
                if index == 0:
                    start_idx += len(conversation['content'])-1 - overlap_half
                else:
                    start_idx += len(conversation['content'])-1 - overlap_half*2
        eff_score[0] = None
        fill_score = eff_score[:len(df)] + [None] * (len(df) - len(eff_score))
        df['audio_text_prob'] = fill_score
        df.to_csv(os.path.join(save_path, f"{number}_audio_text.csv"), index=False, encoding='utf-8-sig')

def main():
    config = load_config("config.yaml")
    os.environ["CUDA_VISIBLE_DEVICES"] = config['audio_text_infer']['cuda_devices']
    model_path = config['audio_text_infer']['infer_model_path']
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, device_map="auto")
    dramas = config['audio_text_infer']['dramas']
    for drama in dramas:
        main_post(drama, config, processor, model)

if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)
    main()
