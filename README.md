# Mirror-BERT

Code repo for the EMNLP 2021 paper: [*Fast, Effective, and Self-Supervised: Transforming Masked Language Models into Universal Lexical and Sentence Encoders*](https://arxiv.org/pdf/2104.08027.pdf).

## Hugginface pretrained models

### Sentence-level
|model | STS avg. |
|------|------|
|[mirror-bert-base-uncased-sentence](https://huggingface.co/cambridgeltl/mirror-bert-base-uncased-sentence)|74.51|
|[mirror-roberta-base-sentence](https://huggingface.co/cambridgeltl/mirror-roberta-base-sentence)|75.08|
|[mirror-bert-base-uncased-sentence-drophead](https://huggingface.co/cambridgeltl/mirror-bert-base-uncased-sentence-drophead)|75.16|
|[mirror-roberta-base-sentence-drophead](https://huggingface.co/cambridgeltl/mirror-roberta-base-sentence-drophead)| 76.67|

(Note that the released models would not replicate the exact numbers in the paper, since the reported numbers in the paper are average of three runs.)


## Evaluate

```python
python evaluation/sent_eval.py \
  --model_dir "cambridgeltl/mirror-roberta-base-sentence-drophead" \
  --agg_mode "cls"
```

Training code and more model weights coming in a few days!

## Citation
```bibtex
@inproceedings{
  liu2021fast,
  title={Fast, Effective and Self-Supervised: Transforming Masked LanguageModels into Universal Lexical and Sentence Encoders},
  author={Liu, Fangyu and Vuli{\'c}, Ivan and Korhonen, Anna and Collier, Nigel},
  booktitle={EMNLP 2021},
  year={2021}
}
```
