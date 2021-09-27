import os
import pickle
import logging
import torch
import numpy as np
import time
from tqdm import tqdm
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoTokenizer, 
    AutoModel, 
)

from .contrastive_learning import *

LOGGER = logging.getLogger()


class MirrorBERT(object):
    """
    Wrapper class for MirrorBERT 
    """

    def __init__(self):
        self.tokenizer = None
        self.encoder = None

    def get_encoder(self):
        assert (self.encoder is not None)

        return self.encoder

    def get_tokenizer(self):
        assert (self.tokenizer is not None)

        return self.tokenizer

    def save_model(self, path, context=False):
        # save bert model, bert config
        self.encoder.save_pretrained(path)

        # save bert vocab
        self.tokenizer.save_pretrained(path)

    def load_model(self, path, max_length=50, lowercase=True, 
            use_cuda=True, return_model=False):
        
        self.tokenizer = AutoTokenizer.from_pretrained(path, 
                use_fast=True, do_lower_case=lowercase)
        self.encoder = AutoModel.from_pretrained(path)
        if use_cuda:
            self.encoder = self.encoder.cuda()
        if not return_model:
            return
        return self.encoder, self.tokenizer
    
    def encode(self, sentences, max_length=50, agg_mode="cls"):
        sent_toks = self.tokenizer.batch_encode_plus(
            list(sentences), 
            max_length=max_length, 
            padding="max_length", 
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")
        sent_toks_cuda = {}
        for k,v in sent_toks.items():
            sent_toks_cuda[k] = v.cuda()
        with torch.no_grad():
            outputs = self.encoder(**sent_toks_cuda, return_dict=True, output_hidden_states=False)
        last_hidden_state = outputs.last_hidden_state

        if agg_mode=="cls":
            query_embed = last_hidden_state[:,0]  
        elif agg_mode == "mean": # including padded tokens
            query_embed = last_hidden_state.mean(1)  
        elif agg_mode == "mean_std":
            query_embed = (last_hidden_state * query_toks['attention_mask'].unsqueeze(-1)).sum(1) / query_toks['attention_mask'].sum(-1).unsqueeze(-1)
        else:
            raise NotImplementedError()
        return query_embed

    def get_embeddings(self, sentences, batch_size=1024, max_length=50, agg_mode="cls"):
        """
        Compute embeddings from a list of sentence.
        """
        embedding_table = []
        with torch.no_grad():
            for start in tqdm(range(0, len(sentences), batch_size)):
                end = min(start + batch_size, len(sentences))
                batch = sentences[start:end]
                batch_embedding = self.encode(batch, max_length=max_length, agg_mode=agg_mode)
                batch_embedding = batch_embedding.cpu()
                embedding_table.append(batch_embedding)
        embedding_table = torch.cat(embedding_table, dim=0)
        return embedding_table


