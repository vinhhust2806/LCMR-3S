#!/bin/bash
set -e
cd ..
GROUP=experiments

#########################################
############# TWITTER #### 
#########################################
################################
python main.py  --config_file configs/combos/clip_roberta.yaml --name fold-0-twitter-ws-128-clip-roberta --group $GROUP  --dataset twitter --fold 0 --window_size 128 --position_embeddings zero --mode run --epochs $EPOCHS --batch_size $BATCH_SIZE
python evaluate.py  --config_file configs/combos/clip_roberta.yaml --name fold-0-twitter-ws-128-clip-roberta --group $GROUP --dataset twitter --fold 0 --window_size 128  --position_embeddings zero --output_dir $GROUP

################################################
################### Reddit ####################
################################################
python main.py  --config_file configs/combos/clip_roberta.yaml --name reddit-ws-128-clip-roberta --group $GROUP --dataset reddit --window_size 128  --position_embeddings zero --mode run --epochs $EPOCHS --batch_size $BATCH_SIZE
python evaluate.py  --config_file configs/combos/clip_roberta.yaml --name reddit-ws-128-clip-roberta --group $GROUP --dataset reddit --window_size 128  --position_embeddings zero  --output_dir $GROUP
