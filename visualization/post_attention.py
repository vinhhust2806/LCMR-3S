import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, '')
import nomenclature
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

def save_text_gradient(grad):
    text_gradients.append(grad)

hook = model.text_projection.register_backward_hook(lambda m, grad_input, grad_output: save_text_gradient(grad_output[0]))
label = {0:"negative", 1:"positive"}

if __name__ == '__main__':
  text_gradients = []

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

    torch.nn.BCEWithLogitsLoss(reduction="none")(output["logits"], torch.tensor([[i['label']]]).cuda()).backward()
    hook.remove()
   
    out = text_gradients[0][0][:len(i["idx"][0]),:].cpu().detach().numpy()
    plt.figure(figsize=(36, 36))
    plt.xlabel("Representation", fontsize = 20)
    plt.ylabel("Post", fontsize = 20)
    plt.title(i["user"][0], fontsize = 20)
    plt.imshow(out, cmap='viridis')
    plt.colorbar(label='Feature Intensity')
    plt.show()



