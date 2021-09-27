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

    def get_dense_encoder(self):
        assert (self.encoder is not None)

        return self.encoder

    def get_dense_tokenizer(self):
        assert (self.tokenizer is not None)

        return self.tokenizer

    def save_model(self, path, context=False):
        # save bert model, bert config
        self.encoder.save_pretrained(path)

        # save bert vocab
        self.tokenizer.save_pretrained(path)
        

    def load_model(self, path, max_length=25, use_cuda=True, lowercase=True):
        self.load_bert(path, max_length, use_cuda)
        
        return self

    def load_bert(self, path, max_length, use_cuda, lowercase=True):
        self.tokenizer = AutoTokenizer.from_pretrained(path, 
                use_fast=True, do_lower_case=lowercase)
        self.encoder = AutoModel.from_pretrained(path)
        if use_cuda:
            self.encoder = self.encoder.cuda()

        return self.encoder, self.tokenizer
    
    def encode(self, sentences, max_length=50, ):
        sent_toks = tokenizer.batch_encode_plus(
            list(sentences), 
            max_length=max_length, 
            padding="max_length", 
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")
        sent_toks_cuda = {}
        for k,v in sent_toks.items():
            sent_toks_cuda[k] = v.cuda()
        return self.encoder.get_embeddings(sent_toks_cuda)
         
