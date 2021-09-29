import os
import csv 
import torch
import pandas
import argparse
import subprocess
from tqdm.auto import tqdm
from zipfile import ZipFile
import numpy as np
from scipy import spatial
from scipy.stats.stats import pearsonr,spearmanr
from transformers import AutoTokenizer, AutoModel

from load_data import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Word/sentence similarity evaluation.')
    parser.add_argument('--model_dir', type=str, default="cambridgeltl/mirror-roberta-base-sentence-drophead")
    parser.add_argument('--agg_mode', type=str, default="cls", help="{cls|mean|mean_std|...}") 
    parser.add_argument('--dataset', type=str, default="sent_all")
    parser.add_argument('--maxlen', type=int, default=64) 
    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--device', type=int, default=0) 


    args = parser.parse_args()

    model_name = args.model_dir 
    agg_mode = args.agg_mode
    device = args.device

    bsz = args.batch_size
    maxlen = args.maxlen
    if args.dataset == "sent_all":
        datasets = ["sts2012", "sts2013", "sts2014", "sts2015", "sts2016", "stsb", "sickr"]
    else:
        datasets = [args.dataset]

    # For sentence: check if STS_data exist
    if not os.path.exists("./data/STS_data"):
        zip_save_path = "./data/STS_data.zip"
        subprocess.run(["wget", "--no-check-certificate", "https://fangyuliu.me/data/STS_data.zip", "-P", "data/"])
        with ZipFile(zip_save_path, "r") as zipIn:
            zipIn.extractall("./data/")
    
    # For word: check if multisimlex exist
    if not os.path.exists("./data/multisimlex"):
        subprocess.run(["mkdir", "./data/multisimlex"])
        subprocess.run(["wget", "--no-check-certificate", "https://multisimlex.com/data/translation.csv", "-P", "./data/multisimlex/"])
        subprocess.run(["wget", "--no-check-certificate", "https://multisimlex.com/data/scores.csv", "-P", "./data/multisimlex/"])

    
    # load the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).cuda(device).eval()

    all_scores = []
    for d in datasets:
        if d == "stsb":
            dataset = load_stsb()
        elif d == "sickr": 
            dataset = load_sickr()
        elif "sts201" in d:
            dataset = load_sts201x(d)
        elif "multisimlex" in d:
            lang = d[-3:]
            dataset = load_multisimlex(lang)
        else:
            raise NotImplementedError()

        sents1, sents2, gold_scores = dataset
        print (d, "size:", len(sents1))

        string_features1, string_features2 = [], []
        for i in tqdm(np.arange(0, len(sents1), bsz)):
            toks = tokenizer.batch_encode_plus(sents1[i:i+bsz], max_length=maxlen,
                                truncation=True, padding="max_length", return_tensors="pt")
            toks_cuda = {k:v.cuda(device) for k,v in toks.items()}
            with torch.no_grad():
                outputs_ = model(**toks_cuda, output_hidden_states=True)
            last_hidden_state = outputs_.last_hidden_state
            pooler_output = outputs_.pooler_output
            if agg_mode == "mean":
                np_feature_mean_tok = last_hidden_state.detach().cpu().numpy().mean(1) # mean-tok over all tokens (incl. padded ones)
            if agg_mode == "mean_std":
                np_feature_mean_tok = ((last_hidden_state * toks_cuda['attention_mask'].unsqueeze(-1)).sum(1) / toks_cuda['attention_mask'].sum(-1).unsqueeze(-1)).cpu()
            elif agg_mode == "cls_pooler":
                np_feature_mean_tok = pooler_output.detach().cpu().numpy() # use [CLS] of pooler
            elif agg_mode == "first_tok":
                np_feature_mean_tok = last_hidden_state.detach().cpu().numpy()[:,1,:]  # first sub-tok
            elif agg_mode == "cls":
                np_feature_mean_tok = last_hidden_state.detach().cpu().numpy()[:,0,:] # use [CLS]
            string_features1.append(np_feature_mean_tok)
        string_features1_stacked =  np.concatenate(string_features1, 0)

        for i in tqdm(np.arange(0, len(sents2), bsz)):
            toks = tokenizer.batch_encode_plus(sents2[i:i+bsz], max_length=maxlen,
                                truncation=True, padding="max_length", return_tensors="pt")
            toks_cuda = {k:v.cuda(device) for k,v in toks.items()}
            with torch.no_grad():
                outputs_ = model(**toks_cuda, output_hidden_states=True)
            last_hidden_state = outputs_.last_hidden_state
            pooler_output = outputs_.pooler_output
            if agg_mode == "mean":
                np_feature_mean_tok = last_hidden_state.detach().cpu().numpy().mean(1) # mean-tok over all tokens (incl. padded ones)
            if agg_mode == "mean_std":
                np_feature_mean_tok = ((last_hidden_state * toks_cuda['attention_mask'].unsqueeze(-1)).sum(1) / toks_cuda['attention_mask'].sum(-1).unsqueeze(-1)).cpu()
            elif agg_mode == "cls_pooler":
                np_feature_mean_tok = pooler_output.detach().cpu().numpy() # use [CLS] of pooler
            elif agg_mode == "first_tok":
                np_feature_mean_tok = last_hidden_state.detach().cpu().numpy()[:,1,:]  # first sub-tok
            elif agg_mode == "cls":
                np_feature_mean_tok = last_hidden_state.detach().cpu().numpy()[:,0,:] # use [CLS]
            string_features2.append(np_feature_mean_tok)

        string_features2_stacked =  np.concatenate(string_features2, 0)

        # compute scores
        bert_sims = []
        for i in range(len(string_features1_stacked)):
            result = 1 - spatial.distance.cosine(string_features1_stacked[i], string_features2_stacked[i])
            bert_sims.append(result)

        score = spearmanr(gold_scores, bert_sims)[0]
        print (d+": %.2f"%(score*100))
        all_scores.append(score)

    print (datasets)
    print (" & ".join(["%.2f"%(item*100) for item in all_scores]))
    print("avg: %.2f"%(np.mean(all_scores)*100))


