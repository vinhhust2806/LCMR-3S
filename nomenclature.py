import yaml
from datasets import *
from models import *
from evaluators import MultimodalEvaluator

import torch

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
}
