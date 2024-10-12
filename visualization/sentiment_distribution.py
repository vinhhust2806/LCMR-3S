import sys
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, '')
import nomenclature
from utils_draw import visualize_sentiment_distribution
from utils import load_args, extract_tweets_from_user

parser = argparse.ArgumentParser(description="Do stuff.")
parser.add_argument("--config_file", type=str, required=True)
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--fold", type=int, default=None)
parser.add_argument("--window_size", type=int, default=None)
parser.add_argument("--position_embeddings", type=str, default=None)
parser.add_argument("--image_embeddings_type", type=str, default=None)
parser.add_argument("--text_embeddings_type", type=str, default=None)
parser.add_argument("--weight", type=str, default=None)
parser.add_argument("--kind", type=str, default='test')

args = parser.parse_args()
args, cfg = load_args(args)

dataset = nomenclature.DATASETS[args.dataset]
test_set = dataset(args, kind=args.kind)
test_set = DataLoader(
    test_set,
    batch_size=1,
    num_workers=1,
    pin_memory=True,
)

model = nomenclature.MODELS[args.model]
model = model(args).cuda()
model.load_state_dict(torch.load(args.weight))

label = {0:"negative", 1:"positive"}

if __name__ == '__main__':
  for i in test_set:
    tweets = extract_tweets_from_user(i['user'][0], label[i['label'][0][0].item()])
    tweets = np.array(tweets)[i['idx']][0]
    for k, post in enumerate(tweets):
       print(f"Post {k}.",post)
    
    i['image_embeddings'] = i['image_embeddings'].cuda()
    i['image_mask'] = i['image_mask'].cuda()
    i['text_embeddings'] = i['text_embeddings'].cuda()
    i['text_mask'] = i['text_mask'].cuda()
    output = model(i)
 
    posts = output['cross'][:,:len(i['idx'][0]),:].cpu().detach().squeeze(0).numpy()
    visualize_sentiment_distribution(posts)
  