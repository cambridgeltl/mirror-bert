"""
This file is downloaded directly from https://github.com/Kirill-Kravtsov/drophead-pytorch/blob/master/drophead.py
Credit goes to: https://github.com/Kirill-Kravtsov
"""


import torch
import torch.nn.functional as F
from transformers import BertModel, RobertaModel, XLMRobertaModel


VALID_CLS = (BertModel, RobertaModel, XLMRobertaModel)


def _drophead_hook(module, input, output):
    """
    Pytorch forward hook for transformers.modeling_bert.BertSelfAttention layer
    """
    if (not module.training) or (module.p_drophead==0):
        return output

    orig_shape = output[0].shape
    dist = torch.distributions.Bernoulli(torch.tensor([1-module.p_drophead]))
    mask = dist.sample((orig_shape[0], module.num_attention_heads))
    mask = mask.to(output[0].device).unsqueeze(-1)
    count_ones = mask.sum(dim=1).unsqueeze(-1)  # calc num of active heads

    self_att_out = module.transpose_for_scores(output[0])
    self_att_out = self_att_out * mask * module.num_attention_heads / count_ones
    self_att_out = self_att_out.permute(0, 2, 1, 3).view(*orig_shape)
    return (self_att_out,) + output[1:]


def valid_type(obj):
    return isinstance(obj, VALID_CLS)


def get_base_model(model):
    """
    Check model type. If correct then return the model itself.
    If not correct then try to find in attributes and return correct type
    attribute if found
    """
    if not valid_type(model):
        attrs = [name for name in dir(model) if valid_type(getattr(model, name))]
        if len(attrs) == 0:
            raise ValueError("Please provide valid model")
        model =  getattr(model, attrs[0])
    return model


def set_drophead(model, p=0.1):
    """
    Adds drophead to model. Works inplace.
    Args:
        model: an instance of transformers.BertModel / transformers.RobertaModel /
            transformers.XLMRobertaModel or downstream model (e.g. transformers.BertForSequenceClassification)
            or any custom downstream model
        p: drophead probability
    """
    if (p < 0) or (p > 1):
        raise ValueError("Wrong p argument")

    model = get_base_model(model)

    for bert_layer in model.encoder.layer:
        if not hasattr(bert_layer.attention.self, "p_drophead"):
            bert_layer.attention.self.register_forward_hook(_drophead_hook)
        bert_layer.attention.self.p_drophead = p
