import csv
import pandas

def load_stsb():
    # STS benchmark (en)
    sts = pandas.read_csv("./data/STS_data/en/sts-b-test.txt", delimiter="\t",
                      names=["genre","filename","year", "?", "score", "sent1", "sent2", "license"],
                      quoting=csv.QUOTE_NONE, encoding='utf-8') 
    return sts["sent1"].tolist(), sts["sent2"].tolist(), sts["score"].tolist()

def load_sts201x(dataset="sts2012"):
    # STS 2012-2016
    year = dataset[3:]
    sts = pandas.read_csv(f"./data/STS_data/en/{year}.test.tsv",
                      delimiter="\t", names=["score","sent1", "sent2"], quoting=csv.QUOTE_NONE, encoding='utf-8')
    sts = sts.dropna()
    return sts["sent1"].tolist(), sts["sent2"].tolist(), sts["score"].tolist()

def load_sickr():
    # SICK-R dataset
    sts = pandas.read_csv("./data/STS_data/en/SICK_annotated.txt", delimiter="\t",
                      quoting=csv.QUOTE_NONE, encoding='utf-8')
    sts = sts[sts["SemEval_set"] == "TEST"]
    return sts["sentence_A"].tolist(), sts["sentence_B"].tolist(), sts["relatedness_score"].tolist()

def load_multisimlex(lang="ENG"):
    # multisimlex dataset
    simlex = pandas.read_csv("./data/multisimlex/scores.csv", delimiter=",")
    simlex_translation = pandas.read_csv("./data/multisimlex/translation.csv", delimiter=",")
    return simlex_translation[f"{lang} 1"].tolist(), simlex_translation[f"{lang} 2"].tolist(), simlex[f"{lang}"].tolist()