#!/usr/bin/env python
import os
import sys
import pdb
import time
import random
import json
import logging
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim import Adam, Adadelta, Adamax, Adagrad, RMSprop, Rprop, SGD
from torch.cuda.amp import autocast, GradScaler
from pytorch_metric_learning import samplers
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import wandb
wandb.init(project="mirror-bert")

# import from local
from src.mirror_bert import MirrorBERT
from src.data_loader import ContrastiveLearningDataset
from src.contrastive_learning import ContrastiveLearningPairwise
from src.drophead import set_drophead

LOGGER = logging.getLogger()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='train Mirror-BERT')

    # Required
    parser.add_argument('--train_dir', type=str, required=True, help='training set directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory for output')

    parser.add_argument('--model_dir', type=str, \
        help='Directory for pretrained model', \
        default="roberta-base")
    parser.add_argument('--max_length', default=50, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--train_batch_size', default=200, type=int)
    parser.add_argument('--epoch', default=3, type=int)
    parser.add_argument('--infoNCE_tau', default=0.04, type=float) 
    parser.add_argument('--agg_mode', default="cls", type=str, help="{cls|mean|mean_std}") 
    parser.add_argument('--use_cuda',  action="store_true")
    parser.add_argument('--save_checkpoint_all', action="store_true")
    parser.add_argument('--checkpoint_step', type=int, default=10000000)
    parser.add_argument('--parallel', action="store_true") 
    parser.add_argument('--amp', action="store_true", \
        help="automatic mixed precision training")
    parser.add_argument('--pairwise', action="store_true", \
        help="if loading pairwise formatted datasets") 
    parser.add_argument('--random_seed', default=42, type=int)

    # data augmentation config
    parser.add_argument('--dropout_rate', default=0.1, type=float) 
    parser.add_argument('--drophead_rate', default=0.0, type=float)
    parser.add_argument('--random_span_mask', default=5, type=int, 
            help="number of chars to be randomly masked on one side of the input") 

    args = parser.parse_args()
    return args

def init_logging():
    LOGGER.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    LOGGER.addHandler(console)


def train(args, data_loader, model, scaler=None, mirror_bert=None, step_global=0):
    LOGGER.info("train!")

    pairwise = args.pairwise
    
    train_loss = 0
    train_steps = 0
    model.cuda()
    model.train()
    for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        model.optimizer.zero_grad()

        batch_x1, batch_x2 = data
        batch_x_cuda1, batch_x_cuda2 = {},{}
        for k,v in batch_x1.items():
            batch_x_cuda1[k] = v.cuda()
        for k,v in batch_x2.items():
            batch_x_cuda2[k] = v.cuda()

        if args.amp:
            with autocast():
                loss = model(batch_x_cuda1, batch_x_cuda2)
        else:
            loss = model(batch_x_cuda1, batch_x_cuda2)  

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(model.optimizer)
            scaler.update()
        else:
            loss.backward()
            model.optimizer.step()

        train_loss += loss.item()
        wandb.log({"Loss": loss.item()})
        train_steps += 1
        step_global += 1

        # save model every K iterations
        if step_global % args.checkpoint_step == 0:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_iter_{step_global}")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            mirror_bert.save_model(checkpoint_dir)
    train_loss /= (train_steps + 1e-9)
    return train_loss, step_global
    
def main(args):
    init_logging()
    print(args)

    torch.manual_seed(args.random_seed) 
    # by default 42 is used, also tried 33, 44, 55
    # results don't seem to change too much
    
    # prepare for output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # load BERT tokenizer, dense_encoder
    mirror_bert = MirrorBERT()
    encoder, tokenizer = mirror_bert.load_model(
        path=args.model_dir,
        max_length=args.max_length,
        use_cuda=args.use_cuda,
        return_model=True
    )

    # adjust dropout rates
    encoder.embeddings.dropout = torch.nn.Dropout(p=args.dropout_rate)
    for i in range(len(encoder.encoder.layer)):
        encoder.encoder.layer[i].attention.self.dropout = torch.nn.Dropout(p=args.dropout_rate)
        encoder.encoder.layer[i].attention.output.dropout = torch.nn.Dropout(p=args.dropout_rate)
        encoder.encoder.layer[i].output.dropout  = torch.nn.Dropout(p=args.dropout_rate)

    # set drophead rate
    if args.drophead_rate != 0:
        set_drophead(encoder, args.drophead_rate)
    
    # load contrastive learning model
    model = ContrastiveLearningPairwise(
        encoder=encoder,
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay,
        use_cuda=args.use_cuda,
        infoNCE_tau=args.infoNCE_tau,
        agg_mode=args.agg_mode,
    )

    if args.parallel:
        model.encoder = torch.nn.DataParallel(model.encoder)
        LOGGER.info("using nn.DataParallel")
    
    def collate_fn_batch_encoding_pairwise(batch):
        sent1, sent2 = zip(*batch)
        sent1_toks = tokenizer.batch_encode_plus(
            list(sent1), 
            max_length=args.max_length, 
            padding="max_length", 
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")
        sent2_toks = tokenizer.batch_encode_plus(
            list(sent2), 
            max_length=args.max_length, 
            padding="max_length", 
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt")
        return sent1_toks, sent2_toks

    train_set = ContrastiveLearningDataset(
        path=args.train_dir,
        tokenizer=tokenizer,
        random_span_mask=args.random_span_mask,
        pairwise=args.pairwise
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=16,
        collate_fn=collate_fn_batch_encoding_pairwise,
        drop_last=True
    )

    # mixed precision training 
    if args.amp:
        scaler = GradScaler()
    else:
        scaler = None

    start = time.time()
    step_global = 0
    for epoch in range(1,args.epoch+1):
        LOGGER.info(f"Epoch {epoch}/{args.epoch}")

        # train
        train_loss, step_global = train(args, data_loader=train_loader, model=model, 
                scaler=scaler, mirror_bert=mirror_bert, step_global=step_global)
        LOGGER.info(f'loss/train_per_epoch={train_loss}/{epoch}')
        
        # save model every epoch
        if args.save_checkpoint_all:
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint_{epoch}")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            mirror_bert.save_model(checkpoint_dir)
        
        # save model last epoch
        if epoch == args.epoch:
            mirror_bert.save_model(args.output_dir)
            
    end = time.time()
    training_time = end-start
    training_hour = int(training_time/60/60)
    training_minute = int(training_time/60 % 60)
    training_second = int(training_time % 60)
    LOGGER.info(f"Training Time!{training_hour} hours {training_minute} minutes {training_second} seconds")
    
if __name__ == '__main__':
    args = parse_args()
    main(args)
