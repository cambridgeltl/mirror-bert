import re
import os
import glob
import numpy as np
import random
import random
import pandas as pd
import json
from torch.utils.data import Dataset
import logging
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

def erase_and_mask(s, tokenizer, mask_len=5):
    """
    Randomly replace a span in input s with "[MASK]".
    """
    if len(s) <= mask_len: return s
    if len(s) < 30: return s # if too short, no span masking
    ind = np.random.randint(len(s)-mask_len)
    left, right = s.split(s[ind:ind+mask_len], 1)
    return " ".join([left, "[MASK]", right]) 
    # I realised that for RoBERTa, it actually should be <MASK> (or tokenizer.mask_token); 
    # but interestingly this doesn't really hurt the model's performance

class ContrastiveLearningDataset(Dataset):
    def __init__(self, path, tokenizer, random_span_mask=0, pairwise=False): 
        with open(path, 'r') as f:
            lines = f.readlines()
        self.sent_pairs = []
        self.pairwise = pairwise

        if self.pairwise: # used for supervised setting
            for line in lines:
                line = line.rstrip("\n")
                try:
                    sent1, sent2 = line.split("||")
                except:
                    continue
                self.sent_pairs.append((sent1, sent2))
        else:
            for i, line in enumerate(lines):
                sent = line.rstrip("\n")
                self.sent_pairs.append((sent, sent))
        self.tokenizer = tokenizer
        self.random_span_mask = random_span_mask
    
    def __getitem__(self, idx):

        sent1 = self.sent_pairs[idx][0]
        sent2 = self.sent_pairs[idx][1]
        if self.random_span_mask != 0:
            sent2 = erase_and_mask(sent2, self.tokenizer, mask_len=int(self.random_span_mask))
        return sent1, sent2

    def __len__(self):
        assert (len(self.sent_pairs) !=0)
        return len(self.sent_pairs)
