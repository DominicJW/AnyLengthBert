import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup


import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric
from  datasets import Dataset as datasetsDataset
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef,accuracy_score

import os
import copy
import random
import math
import string

criterion = nn.BCEWithLogitsLoss() 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer_name='albert-base-v2', max_length=128,with_labels = True):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.data = df
        self.max_length = max_length
        self.chunked_samples = []
        self.make_chunks()
        self.with_labels = with_labels
    def make_chunks(self):
        for i in range(len(self.data)):
            self.chunked_samples.append(self.make_chunk(i))

    def __getitem__(self, idx): #last for number of chunks
        # if self.with_labels:
        #     return [*self.chunked_samples[idx] , self.data.iloc[idx]["label"],len(self.chunked_samples[idx][0])]
        # else:
        #     return [*self.chunked_samples[idx] , 0, len(self.chunked_samples[idx][0])]
        if self.with_labels:
            return [*self.chunked_samples[idx] , self.data.iloc[idx]["label"],1]
        else:
            return [*self.chunked_samples[idx] , 0, 1]


            



    def preprocess_claim(self,claim):
        if claim.startswith("We should "):
            claim = claim[10:]
        return self.preprocess_text(claim)

    def preprocess_evidence(self,evidence):
        evidence = evidence.replace("[REF]"," ")
        return self.preprocess_text(evidence)

    def preprocess_text(self,text):
        text = text.lower()
        punctuation = string.punctuation
        text = ''.join(char for char in text if char not in punctuation)
        return text

    def make_chunk(self, idx):

        row = self.data.iloc[idx]
        claim = row["Claim"]
        evidence = row["Evidence"]

        try:
            claim = self.preprocess_claim(claim)
        except:
            claim = " "
        try:
            evidence = self.preprocess_evidence(evidence)
        except:
            evidence = " "

        claim_tokens = self.tokenizer.encode_plus(claim, add_special_tokens=False, return_tensors='pt', truncation=False)
        evidence_tokens = self.tokenizer.encode_plus(evidence, add_special_tokens=False, return_tensors='pt', truncation=False)

        evidence_input_ids = evidence_tokens['input_ids'][0]
        claim_input_ids = claim_tokens['input_ids'][0]

        if len(evidence_input_ids) + len(claim_input_ids) + 3 <= self.max_length:
            chunk_input_ids = [self.tokenizer.cls_token_id] + claim_input_ids.tolist()  + [self.tokenizer.sep_token_id]  + evidence_input_ids.tolist() + [self.tokenizer.sep_token_id]
            attention_mask = [1] * len(chunk_input_ids)
            padding_length = self.max_length - len(chunk_input_ids)
            chunk_input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            ids_chunks = torch.stack([torch.tensor(chunk_input_ids)])
            attention_mask_chunks = torch.stack([torch.tensor(attention_mask)])
            return ids_chunks, attention_mask_chunks

        ids_chunks = []
        attention_mask_chunks = []

        claim_len = (
            min(
                max(
                    0.6,
                    1- len(evidence_input_ids)/(self.max_length - 3)
                    )
                ,len(claim_input_ids)/(self.max_length - 3)))



        claim_tokens_per_chunk = math.floor(claim_len*(self.max_length - 3))
        evidence_tokens_per_chunk = (self.max_length - 3) - claim_tokens_per_chunk

        overlap = 5
        stride = max(evidence_tokens_per_chunk - overlap,1)
        for i in range(0,len(evidence_input_ids),stride):

            chunk_input_ids = [self.tokenizer.cls_token_id] + claim_input_ids.tolist()[:claim_tokens_per_chunk] + [self.tokenizer.sep_token_id]  + evidence_input_ids[i:i+evidence_tokens_per_chunk].tolist() + [self.tokenizer.sep_token_id]
            attention_mask = [1] * len(chunk_input_ids)
            padding_length = self.max_length - len(chunk_input_ids)
            chunk_input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            ids_chunks.append(torch.tensor(chunk_input_ids, dtype=torch.long))
            attention_mask_chunks.append(torch.tensor(attention_mask, dtype=torch.long))

        try:
            ids_chunks = torch.stack(ids_chunks)
        except:
            print(f"evidence_tokens_per_chunk {evidence_tokens_per_chunk}" )
            print(f"len(evidence_input_ids) {len(evidence_input_ids)}" )
            print(f"stride {stride}" )
            print(evidence)
            raise Exception()

        attention_mask_chunks = torch.stack(attention_mask_chunks)

        return ids_chunks, attention_mask_chunks

    def __len__(self):
        return len(self.chunked_samples)



def my_collate(batch):
    ids_chunks, attention_mask_chunks, labels, chunk_length = zip(*batch)
    max_num_chunks = max(len(chunks) for chunks in ids_chunks)
    padded_ids_chunks = []
    padded_attention_mask_chunks = []
    for ids_chunk, attention_chunk in zip(ids_chunks, attention_mask_chunks):
        num_padding = max_num_chunks - len(ids_chunk)
        padded_ids = torch.cat((ids_chunk, torch.zeros(num_padding, ids_chunk.size(1), dtype=ids_chunk.dtype)), dim=0)
        padded_attention = torch.cat((attention_chunk, torch.zeros(num_padding, attention_chunk.size(1), dtype=attention_chunk.dtype)), dim=0)
        padded_ids_chunks.append(padded_ids)
        padded_attention_mask_chunks.append(padded_attention)

    batch_ids_chunks = torch.stack(padded_ids_chunks)
    batch_attention_mask_chunks = torch.stack(padded_attention_mask_chunks)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    chunk_length_tensor = torch.tensor(chunk_length, dtype=torch.long)
    return batch_ids_chunks, batch_attention_mask_chunks, labels_tensor, chunk_length_tensor


class BlankSched:
    @classmethod
    def step(cls):
        return 1


class MaskedGlobalAvgPool1d(nn.Module):
    def __init__(self):
        super(MaskedGlobalAvgPool1d, self).__init__()

    def forward(self, x, mask):
        mask = mask
        x = torch.sum(x,dim = 1)
        batch_wise_non_masked_count = torch.sum(x,dim=1).unsqueeze(1)
        x = x / batch_wise_non_masked_count.clamp(min = 1)
        return x


class MaskedGlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(MaskedGlobalMaxPool1d, self).__init__()

    def forward(self, x):
        x = torch.max(x,dim = 1).values
        return x

class SentencePairClassifier(nn.Module):
    def __init__(self,bert_model="albert-base-v2", hidden_size =768 ,freeze_bert = False,dropout_p= 0.1):
        super(SentencePairClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model)
        self.hidden_size = hidden_size
        self.cls_layer = nn.Sequential(
            nn.Linear(768,1)
        )
        self.layer_norm = nn.LayerNorm(768) 
        self.dropout = nn.Dropout(p=dropout_p)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        self.global_avg_pool = MaskedGlobalAvgPool1d()
        self.global_max_pool = MaskedGlobalMaxPool1d()

    def set_freeze_bert(self,freeze):
        self.freeze_bert = freeze
        if self.freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        else:
            for p in self.bert.parameters():
                p.requires_grad = True

    def set_freeze_layer_in_albert(self,i,to_freeze=False):
        layer_regex = f"albert.encoder.albert_layer_groups.0.albert_layers.{i}."
        for name, param in self.bert.named_parameters():
            print(name)
            if layer_regex in name:
                param.requires_grad = True
                print(f"Unfroze layer: {name}")

    def set_bert_frozen(self,frozen = False):
        if frozen:
            for p in self.bert.parameters():
                p.requires_grad = False
            print("bert frozen")
        else:
            for p in self.bert.parameters():
                p.requires_grad = True
            print("bert unfrozen")

    def set_cls_frozen(self,frozen = False):
        for p in self.cls_layer.parameters():
            p.requires_grad = not frozen
        print(f"cls layer {(not frozen) *'un'}frozen")

    @autocast(enabled = True)
    def forward(self, input_ids,attention_masks,new_num_chunks = [1]):
        batch_size, num_chunks, chunk_len = input_ids.size()
        bert_embeds = []
        masks = []
        chunk_only_mask = torch.any(attention_masks != 0,dim=2)
        for i,chunk in enumerate(range(int(max(new_num_chunks)))):
            mask = attention_masks[:,chunk,:]
            outs = self.bert(input_ids=input_ids[:,chunk,:],attention_mask=mask)
            masks.append(mask)
            bert_embeds.append(outs["pooler_output"].unsqueeze(1))

        bert_embeds = torch.concat(bert_embeds,dim=1)
        masks = torch.concat(masks,dim=1)
    
        pooling_out_avg = self.global_avg_pool(bert_embeds,chunk_only_mask)
        pooling_out_max = self.global_max_pool(bert_embeds)

        pooling_joined = torch.stack((pooling_out_avg,pooling_out_max),dim = 1)
        pooling_joined = self.layer_norm(torch.sum(pooling_joined,dim = 1)) 

        logits = self.cls_layer(self.dropout(pooling_joined))

        return logits


def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_probs_from_logits(logits):
    """
    Converts a tensor of logits into an array of probabilities by applying the sigmoid function
    """
    probs = torch.sigmoid(logits.unsqueeze(-1))
    return probs.detach().cpu().numpy()

def evaluate_loss(net, dataloader):
    net.eval()
    mean_loss = 0
    count = 0
    with torch.no_grad():
        all_predictions = []
        all_labels = []
        for it, (seq, attn_masks, labels,lengths) in enumerate(dataloader):
            seq, attn_masks, labels = seq.to(device), attn_masks.to(device), labels.to(device)
            logits = net(seq, attn_masks,lengths)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            count += 1
            probs = get_probs_from_logits(logits)

            all_predictions.extend((probs.flatten()>0.5).astype("uint8"))
            all_labels.extend(labels.flatten().to("cpu"))

    accuracy = accuracy_score(all_labels,all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    mcc = matthews_corrcoef(all_labels, all_predictions)
    accuracy = accuracy_score(all_labels,all_predictions)
    return (mean_loss / count), accuracy, precision,recall,f1,mcc


def test_prediction(net, device, dataloader, with_labels=True, result_file="results/output.txt"):
    """
    Predict the probabilities on a dataset with or without labels and print the result in a file
    """
    net.eval()
    w = open(result_file, 'w')
    probs_all = []

    with torch.no_grad():
        if with_labels:
            for seq,attn_masks,label,num_chunks in tqdm(dataloader): ##dataloader should give dummy labels I have no idea why not?
                seq, attn_masks = seq.to(device), attn_masks.to(device)
                num_chunks = num_chunks.to(device)
                logits = net(seq, attn_masks,num_chunks)
                probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
                probs_all += probs.tolist()
        else:
            for seq,attn_masks,label,num_chunks in tqdm(dataloader): ##dataloader should give dummy labels I have no idea why not?
                seq, attn_masks = seq.to(device), attn_masks.to(device)
                num_chunks = num_chunks.to(device)
                logits = net(seq, attn_masks,num_chunks)
                probs = get_probs_from_logits(logits.squeeze(-1)).squeeze(-1)
                probs_all += probs.tolist()

    w.writelines(str(prob)+'\n' for prob in probs_all)
    w.close()


