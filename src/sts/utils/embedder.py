"""
Following source code of paperwork
"UIT-HSE at WNUT-2020 Task 2: Exploiting CT-BERT for
Identifying COVID-19 Information on the Twitter Social Network"
- Authors: Khiem Tran, Hao Phan (Student БПМИ208 HSE), Kiet Nguyen, Ngan Luu Thuy Nguyen
- Paper link: https://aclanthology.org/2020.wnut-1.53/
- Source link: https://github.com/haophancs/transformers-exptool
"""

import torch
import gc
import os
import torch
import random
import numpy as np
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
default_device = torch.device(os.environ['DEFAULT_DEVICE'])


class TransformersEmbedder:
    @staticmethod
    def extract_features(
            text, pretrained_model,
            tokenizer, encode_config,
            device=default_device, return_tensors='pt',
            random_state=42):
        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        torch.cuda.manual_seed_all(random_state)

        encoded_item = tokenizer.encode_plus(text, **encode_config)
        input_ids = encoded_item['input_ids'].to(device)
        attention_mask = encoded_item['attention_mask'].to(device)
        pretrained_model.eval()
        with torch.no_grad():
            features = pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state, pooled_output, hidden_states = features
        hidden_states = torch.stack(hidden_states).detach().cpu()
        last_hidden_state = last_hidden_state.detach().cpu()
        pooled_output = pooled_output.detach().cpu()
        if return_tensors != 'pt':
            last_hidden_state = last_hidden_state.detach().cpu().numpy()
            pooled_output = pooled_output.numpy()
            hidden_states = hidden_states.numpy()
        del input_ids, attention_mask, pretrained_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return last_hidden_state, pooled_output, hidden_states

    @staticmethod
    def last_layer_features(
            text, pretrained_model,
            tokenizer, encode_config,
            device=default_device, return_tensors='pt',
            random_state=42):
        last_hidden_state = TransformersEmbedder.extract_features(text=text,
                                                                  pretrained_model=pretrained_model,
                                                                  tokenizer=tokenizer,
                                                                  encode_config=encode_config,
                                                                  device=device,
                                                                  return_tensors='pt',
                                                                  random_state=random_state)[0].squeeze()
        if return_tensors != 'pt':
            last_hidden_state = last_hidden_state.detach().cpu().numpy()
        return last_hidden_state

    @staticmethod
    def text_vector(
            text, pretrained_model,
            tokenizer, encode_config,
            device=default_device, return_tensors='pt',
            random_state=42):
        hidden_states = TransformersEmbedder.extract_features(text=text,
                                                              pretrained_model=pretrained_model,
                                                              tokenizer=tokenizer,
                                                              encode_config=encode_config,
                                                              device=device,
                                                              return_tensors='pt',
                                                              random_state=random_state)[2]
        token_embeddings = torch.squeeze(hidden_states, dim=1)
        token_embeddings = token_embeddings.permute(1, 0, 2)
        token_vecs_sum = []
        for token in token_embeddings:
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vecs_sum.append(sum_vec)
        token_vecs = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vecs, dim=0)
        if return_tensors != 'pt':
            sentence_embedding = sentence_embedding.detach().cpu().numpy()
        return sentence_embedding
