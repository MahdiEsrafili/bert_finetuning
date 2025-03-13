import json

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import  BertTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from evaluate import load


metric = load('./cross_entropy_loss')

train_data = pd.read_csv("<path to train file>")
dev_data = pd.read_csv("<path to dev file>")
test_data = pd.read_csv("<path to test file>")

print('training data shape: ', train_data.shape)
print('eval_data  shape: ', dev_data.shape)
print('test_data  shape: ', test_data.shape)

# Encode categories and convert them to integer numbers
encode_dict = {}

def encode_cat(x):
    if x not in encode_dict.keys():
        encode_dict[x]=len(encode_dict)
    return encode_dict[x]

train_data['encoded_cat'] = train_data['category'].apply(lambda x: encode_cat(x))
dev_data['encoded_cat'] = dev_data['category'].apply(lambda x: encode_cat(x))
test_data['encoded_cat'] = test_data['category'].apply(lambda x: encode_cat(x))

with open('encode_dict.json', 'w', encoding='utf-8') as f:
    json.dump(encode_dict, f, ensure_ascii=False)


MAX_LEN = 512
TRAIN_BATCH_SIZE = 24
VALID_BATCH_SIZE = 4

tokenizer = BertTokenizer.from_pretrained('distilbert/distilbert-base-multilingual-cased')

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        paragraph = str(self.data.text[index])
        paragraph = " ".join(paragraph.split())
        inputs = self.tokenizer.encode_plus(
            paragraph,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        item = {key: torch.tensor(inputs[key]) for key in inputs}
        item['labels'] = torch.tensor(self.data.encoded_cat[index], dtype=torch.long)
        return item
    
    def __len__(self):
        return self.len

training_set = TextDataset(train_data, tokenizer, MAX_LEN)
eval_set = TextDataset(dev_data, tokenizer, MAX_LEN)
testing_set = TextDataset(test_data, tokenizer, MAX_LEN)


# Load ParsBERT pre-trained model with the different head that is appropriate for sequence classification
model = AutoModelForSequenceClassification.from_pretrained('distilbert/distilbert-base-multilingual-cased', num_labels=len(encode_dict))

args = TrainingArguments(
    #persian fars text classification finetuned
    f"finetuned",
    overwrite_output_dir=True,
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    save_total_limit=3,
    learning_rate=3e-5,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=VALID_BATCH_SIZE,
    num_train_epochs=20,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='cross_entropy_loss',
    push_to_hub=False,
    report_to="none"
)

def compute_metrics_mathewss(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # predictions = np.argmax(predictions, axis=1)
    return metric.compute(prediction_scores=predictions, references=labels)

trainer = Trainer(
    model,
    args,
    train_dataset=training_set,
    eval_dataset=eval_set,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()