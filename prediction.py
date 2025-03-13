import os
import re
import json
import glob
import pickle

from tqdm import tqdm
import pandas as pd
import hazm
from pandarallel import pandarallel
from cleantext import clean
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

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

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def cleaning(text):
    text = text.strip()
    text = re.sub("\@\w+", "", text)
    # regular cleaning
    text = clean(text,
        fix_unicode=True,
        to_ascii=False,
        lower=True,
        no_line_breaks=True,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=True,
        no_digits=True,
        no_currency_symbols=True,
        no_punct=True,
        replace_with_url="",
        replace_with_email="",
        replace_with_phone_number="",
        replace_with_number="",
        replace_with_digit="",
        replace_with_currency_symbol="",
    )
    # cleaning htmls
    text = cleanhtml(text)
    # normalizing
    normalizer = hazm.Normalizer()
    text = normalizer.normalize(text)
    # removing wierd patterns
    wierd_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u'\U00010000-\U0010ffff'
        u"\u200d"
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\u3030"
        u"\ufe0f"
        u"\u2069"
        u"\u2066"
        # u"\u200c"
        u"\u2068"
        u"\u2067"
        "]+", flags=re.UNICODE)

    text = wierd_pattern.sub(r'', text)
    # removing extra spaces, hashtags
    text = re.sub("#", "", text)
    text = re.sub("\s+", " ", text)
    return text


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
    
