import json
import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base').to('cuda')

name = 'palm'
jsonl_path = f'./{name}/train.jsonl' 
features = []
texts = []
with open(jsonl_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        texts.append(data['text'])  

model.eval()
with torch.no_grad():
    for text in tqdm(texts):
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to('cuda')
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :].squeeze().cpu().numpy() 
        features.append(cls_embedding)

features_np = np.array(features)
np.savez(f'./{name}_features.npz', roberta_features=features_np)