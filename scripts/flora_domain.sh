#!/bin/bash

cd ..

# custom config
DATA="DATA/TO/PATH"
MODEL=FLORA
TRAINER=FLORA
PRETRAINED=True
LR=0.001
GAMMA=1
USERS=8
FRAC=1
ROUND=50
CFG=vit_b16
IID=False
USEALL=True
BETA=0.5
DATASET=$1
SEED=$2
PEFT=$3
TAU=$4
LORA_ENCODER=$5
RANK=2

for DATASET in ${DATASET}
do
  for PARTITION in noniid-labeldir
  do
    for SEED in ${SEED}
    do
      DIR=output/${DATASET}_${PARTITION}_beta${BETA}/${MODEL}_${TRAINER}_csc${CSC}/iid_${IID}_${USERS}users_${FRAC}frac_lr${LR}_${ROUND}round_seed${SEED}/${PEFT}_tau${TAU}_${LORA_ENCODER}_rank${RANK}
      if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
      else
        python federated_main.py \
        --root ${DATA} \
        --model ${MODEL} \
        --seed ${SEED} \
        --tau ${TAU} \
        --peft ${PEFT} \
        --lora_encoder ${LORA_ENCODER} \
        --lora_rank ${RANK} \
        --num_users ${USERS} \
        --frac ${FRAC} \
        --lr ${LR} \
        --gamma ${GAMMA} \
        --trainer ${TRAINER} \
        --round ${ROUND} \
        --partition ${PARTITION} \
        --beta ${BETA} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/vit_b16.yaml \
        --output-dir ${DIR}
      fi
    done
  done
done
