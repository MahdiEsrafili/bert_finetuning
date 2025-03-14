import os
import re
import json
import glob
import pickle

from tqdm import tqdm
import pandas as pd
from pandarallel import pandarallel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from cleaning import *
MAX_LEN = 512

pandarallel.initialize()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

with open('encode_dict.json', 'r', encoding='utf-8') as f:
  label2id = json.load(f)

id2label = {v: k for k, v in label2id.items()}
labels = list(label2id.keys())

print(f'label2id: {label2id}')
print(f'id2label: {id2label}')

tokenizer = AutoTokenizer.from_pretrained("./finetuned/checkpoint-6148")

def prepare_text(texts): 
#    text = cleaning(text)
    encoding = tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LEN,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt')

    inputs = {
        'comments': texts,
        'input_ids': encoding['input_ids'], #.flatten(),
        'attention_mask': encoding['attention_mask'], #.flatten(),
        'token_type_ids': encoding['token_type_ids'] #.flatten(),
    }
    return inputs


pt_model = AutoModelForSequenceClassification.from_pretrained("finetuned/checkpoint-6148")
pt_model = pt_model.to(device)

def predict(texts):
    text_inputs = prepare_text(texts)
    sample_data_comment = text_inputs['comments']
    sample_data_input_ids = text_inputs['input_ids'] #.unsqueeze(0)
    sample_data_attention_mask = text_inputs['attention_mask'] #.unsqueeze(0)
    sample_data_token_type_ids = text_inputs['token_type_ids'] #.unsqueeze(0)
    sample_data_input_ids = sample_data_input_ids.to(device)
    sample_data_attention_mask = sample_data_attention_mask.to(device)
    sample_data_token_type_ids = sample_data_token_type_ids.to(device)
    with torch.no_grad():
        outputs = pt_model(sample_data_input_ids, sample_data_attention_mask, sample_data_token_type_ids)
        outputs = outputs.logits
        _, preds = torch.max(outputs, dim=1)
    
    outputs = torch.softmax(outputs, dim=1)
    outputs = outputs.cpu().numpy()
   
    return outputs

def predict_with_class_names(texts):
    text_inputs = prepare_text(texts)
    sample_data_comment = text_inputs['comments']
    sample_data_input_ids = text_inputs['input_ids'] #.unsqueeze(0)
    sample_data_attention_mask = text_inputs['attention_mask'] #.unsqueeze(0)
    sample_data_token_type_ids = text_inputs['token_type_ids'] #.unsqueeze(0)
    sample_data_input_ids = sample_data_input_ids.to(device)
    sample_data_attention_mask = sample_data_attention_mask.to(device)
    sample_data_token_type_ids = sample_data_token_type_ids.to(device)
    with torch.no_grad():
        outputs = pt_model(sample_data_input_ids, sample_data_attention_mask, sample_data_token_type_ids)
        outputs = outputs.logits
        _, preds = torch.max(outputs, dim=1)
    
    outputs = torch.softmax(outputs, dim=1)
    outputs = outputs.cpu().numpy()
    predicted_dict = []
    for i, preds in enumerate(outputs): 
        obj = [{id2label[idx] :j for idx, j in enumerate(preds)}]
        predicted_dict.append(obj)
    return predicted_dict

files = glob.glob('*.json')

for f in tqdm(files):
    try:
        out_file = os.path.join('preds', f.split('/')[-1].replace('.json', '.pkl'))
        if os.path.exists(out_file): continue
        with open(f, 'r', encoding='utf-8') as file:
            data = json.load(file)
        messages = data.get('latest_messages:', None) or data.get('latest_messages', None)
        if messages is None: continue
        messages.append((data.get('title', '') + ' . ' + data.get('bio', '')))
        df = pd.DataFrame({'latest_messages': messages})
        df['clean'] = df['latest_messages'].parallel_apply(cleaning)
        texts = df['clean'].tolist()
        texts = [t for t in texts if len(t.split())>=2]
        text_embd = predict(texts)
        with open(out_file, 'wb') as file: 
            pickle.dump(text_embd, file)
    except Exception as e:
        print(f)
        print(e)
        
# messages = [
#     "سلام امروز در مورد بیمه می خوام حرف بزنم",
#     "امروز پرسپولیس به تراکتور باخت",
#     "ما همیشه باید پیرو امامان و پیامبران باشیم"
# ]
# try:
#     df = pd.DataFrame({'latest_messages': messages})
#     df['clean'] = df['latest_messages'].parallel_apply(cleaning)
#     texts = df['clean'].tolist()
#     texts = [t for t in texts if len(t.split())>=2]
#     text_embd = predict_with_class_names(texts)
#     # print(text_embd)
#     with open('sample_out.json', 'w', encoding='utf-8') as f:
#         f.write(str(text_embd))
#     # embds = [e for e in text_embd if e!=[]]
#     # with open(out_file, 'wb') as file: 
#     #     pickle.dump(text_embd, file)
# except Exception as e:
#     print(f)
#     print(e)
    
