import torch
from models import *
from datasets import *
from evaluators import MultimodalEvaluator

device = torch.device("cuda")

DATASETS = {
    #"reddit": RedditDataset,
    "twitter": TwitterDataset,
}

EVALUATORS = {
    "multimodal-evaluator": MultimodalEvaluator,
}

MODELS = {
    "LCMR-3S": LCMR3S,
    "SingleModal": SingleModal,
    "tlstm": TimeLSTM
}
