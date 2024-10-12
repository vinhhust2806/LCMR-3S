import sys
import torch
import argparse
from torch.utils.data import DataLoader

sys.path.insert(0, '')
import nomenclature
from utils import load_args
from utils_draw import visualize_state_space

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
parser.add_argument("--type", type=str, default='Depressed')

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

label = {"Depressed":0, "Non-Depressed":1}

if __name__ == '__main__':
  before_ssm = []
  after_ssm = []
  
  for i in test_set:
    if i['label'] == label[args.type]:
      i['image_embeddings'] = i['image_embeddings'].cuda()
      i['image_mask'] = i['image_mask'].cuda()
      i['text_embeddings'] = i['text_embeddings'].cuda()
      i['text_mask'] = i['text_mask'].cuda()
      output = model(i)
    
      before_ssm.append(output['cross'][0].mean(dim=0).unsqueeze(0))
      after_ssm.append(output['ssm'])
    
  before_ssm = torch.cat(before_ssm, dim=0).cpu().detach().numpy()
  after_ssm = torch.cat(after_ssm, dim=0).cpu().detach().numpy()
  visualize_state_space(before_ssm, after_ssm, args.type)

