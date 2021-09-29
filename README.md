# Mirror-BERT

<img align="right" width="400"  src="https://production-media.paperswithcode.com/methods/cd18d6ac-ca08-4fdb-bc69-69e4551372d1.png">

Code repo for the EMNLP 2021 paper: <br>
[*Fast, Effective, and Self-Supervised: Transforming Masked Language Models into Universal Lexical and Sentence Encoders*](https://arxiv.org/pdf/2104.08027.pdf)<br>
by [Fangyu Liu](http://fangyuliu.me/about.html), [Ivan Vulić](https://sites.google.com/site/ivanvulic/), [Anna Korhonen](https://sites.google.com/site/annakorhonen/), and [Nigel Collier](https://sites.google.com/site/nhcollier/). 

Mirror-BERT is a contrastive learning method that converts pretrained language models (PLMs) into universal text encoders. It takes a PLM and a txt file containing raw text as input, and output a strong text embedding model, in just 20-30 seconds. It works well for not only sentence, but also word and phrase representation learning.

## Hugginface pretrained models

|model | STS avg. |
|------|------|
|[mirror-bert-base-uncased-sentence](https://huggingface.co/cambridgeltl/mirror-bert-base-uncased-sentence)|74.51|
|[mirror-roberta-base-sentence](https://huggingface.co/cambridgeltl/mirror-roberta-base-sentence)|75.08|
|[mirror-bert-base-uncased-sentence-drophead](https://huggingface.co/cambridgeltl/mirror-bert-base-uncased-sentence-drophead)|75.16|
|[mirror-roberta-base-sentence-drophead](https://huggingface.co/cambridgeltl/mirror-roberta-base-sentence-drophead)| 76.67|

(Note that the released models would not replicate the exact numbers in the paper, since the reported numbers in the paper are average of three runs.)

## Train
```bash
./mirror_scripts/mirror_sentence_bert.sh 0,1
```
where `0,1` are GPU indices. This script should complete in 20-30 seconds on two NVIDIA 2080Ti/3090 GPUs. If you encounter out-of-memory error, consider reducing `max_length` in the script.

For training with your custom corpus, simply set `train_dir` in the script to your own txt file (one sentence per line). When you do have raw sentences from your target domain, we recommend you always use the in-domain data for optimal performance.

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
```bash
python evaluation/sent_eval.py \
  --model_dir "cambridgeltl/mirror-roberta-base-sentence-drophead" \
  --agg_mode "cls"
```

Training code and model weights for lexical-level tasks are coming in a few days!

## Citation
```bibtex
@inproceedings{liu2021fast,
  title={Fast, Effective, and Self-Supervised: Transforming Masked Language Models into Universal Lexical and Sentence Encoders},
  author={Liu, Fangyu and Vuli{\'c}, Ivan and Korhonen, Anna and Collier, Nigel},
  booktitle={EMNLP 2021},
  year={2021}
}
```
