# Mirror-BERT

<img align="right" width="400"  src="https://production-media.paperswithcode.com/methods/cd18d6ac-ca08-4fdb-bc69-69e4551372d1.png">

**UPDATE**: see a follow-up work [**_Trans-Encoder_ (ICLR'22)**](https://github.com/amzn/trans-encoder), a SotA unsupervised model for STS.

Code repo for the **EMNLP 2021** paper: <br>
[***Fast, Effective, and Self-Supervised: Transforming Masked Language Models into Universal Lexical and Sentence Encoders***](https://arxiv.org/pdf/2104.08027.pdf)<br>
by [Fangyu Liu](http://fangyuliu.me/about.html), [Ivan Vulić](https://sites.google.com/site/ivanvulic/), [Anna Korhonen](https://sites.google.com/site/annakorhonen/), and [Nigel Collier](https://sites.google.com/site/nhcollier/). 

Mirror-BERT is an unsupervised contrastive learning method that converts pretrained language models (PLMs) into universal text encoders. It takes a PLM and a txt file containing raw text as input, and output a strong text embedding model, in just 20-30 seconds. It works well for not only sentence, but also word and phrase representation learning.


## Huggingface pretrained models

Sentence enocders:
|model | STS avg. |
|------|------|
|baseline: sentence-bert (supervised)| 74.89 |
|[mirror-bert-base-uncased-sentence](https://huggingface.co/cambridgeltl/mirror-bert-base-uncased-sentence)|74.51|
|[mirror-roberta-base-sentence](https://huggingface.co/cambridgeltl/mirror-roberta-base-sentence)|75.08|
|[mirror-bert-base-uncased-sentence-drophead](https://huggingface.co/cambridgeltl/mirror-bert-base-uncased-sentence-drophead)|75.16|
|[mirror-roberta-base-sentence-drophead](https://huggingface.co/cambridgeltl/mirror-roberta-base-sentence-drophead)| **76.67** |

Word encoder:
|model | Multi-SimLex (ENG)|
|------|--------|
|baseline: fasttext| 52.80 |
|[mirror-bert-base-uncased-word](https://huggingface.co/cambridgeltl/mirror-bert-base-uncased-word)| **55.60** |

(Note that the released models would not replicate the exact numbers in the paper, since the reported numbers in the paper are average of three runs.)

## Train
For training sentence representations:
```bash
>> ./mirror_scripts/mirror_sentence_bert.sh 0,1
```
where `0,1` are GPU indices. This script should complete in 20-30 seconds on two NVIDIA 2080Ti/3090 GPUs. If you encounter out-of-memory error, consider reducing `max_length` in the script. Scripts for replicating other models are availible in `mirror_scripts/`.

**Custom data:** For training with your custom corpus, simply set `--train_dir` in the script to your own txt file (one sentence per line). When you do have raw sentences from your target domain, we recommend you always use the in-domain data for optimal performance. E.g., if you aim to create a conversational encoder, sample 10k utterances to train your model!

**Supervised training:** Organise your training data in the format of `text1||text2` and store them one pair per line in a txt file. Then turn on the `--pairwise` option. `text1` and `text2` will be regarded as a positive pair in contrastive learning. You can be creative in finding such training pairs and it would be the best if they are from your application domain. E.g., to build an e-commerce QA encoder, the `question||answer` pairs from the [Amazon quesrion-answer dataset](https://jmcauley.ucsd.edu/data/amazon/qa/) could work quite well. Example training script: [`mirror_scripts/mirror_sentence_roberta_supervised_amazon_qa.sh`](https://github.com/cambridgeltl/mirror-bert/blob/main/mirror_scripts/mirror_sentence_roberta_supervised_amazon_qa.sh). Note that when tuned on your in-domain data, you shouldn't expect the model to be good at STS. Instead, the models need to be evaluated on your in-domain task.

**Word-level training:** Use [`mirror_scripts/mirror_word_bert.sh`](https://github.com/cambridgeltl/mirror-bert/blob/main/mirror_scripts/mirror_word_bert.sh). 

## Encode 
It's easy to compute your own sentence embeddings:
```python
from src.mirror_bert import MirrorBERT

model_name = "cambridgeltl/mirror-roberta-base-sentence-drophead"
mirror_bert = MirrorBERT()
mirror_bert.load_model(path=model_name, use_cuda=True)

embeddings = mirror_bert.get_embeddings([
    "I transform pre-trained language models into universal text encoders.",
], agg_mode="cls")
print (embeddings.shape)
```

## Evaluate
Evaluate sentence representations:
```bash
>> python evaluation/eval.py \
	--model_dir "cambridgeltl/mirror-roberta-base-sentence-drophead" \
	--agg_mode "cls" \
	--dataset sent_all
```

Evaluate word representations:
```bash
>> python evaluation/eval.py \
	--model_dir "cambridgeltl/mirror-bert-base-uncased-word" \
	--agg_mode "cls" \
	--dataset multisimlex_ENG
```
To test models on other languages, replace `ENG` to your custom languages. See [here](https://multisimlex.com/) for all supported languages on Multi-SimLex.


## Citation
```bibtex
@inproceedings{liu-etal-2021-fast,
    title = "Fast, Effective, and Self-Supervised: Transforming Masked Language Models into Universal Lexical and Sentence Encoders",
    author = "Liu, Fangyu  and
      Vuli{\'c}, Ivan  and
      Korhonen, Anna  and
      Collier, Nigel",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.109",
    pages = "1442--1459",
}
```
