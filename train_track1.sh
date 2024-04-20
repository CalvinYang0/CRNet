#!/bin/bash
echo "Start to train the model...."
dataroot=""  # including 'Train' and 'NTIRE_Val' floders

device=''
name=""

build_dir="./ckpt/"$name

if [ ! -d "$build_dir" ]; then
        mkdir $build_dir
fi

LOG=./ckpt/$name/`date +%Y-%m-%d-%H-%M-%S`.txt

python train.py \
    --dataset_name bracketire    --model cat        --name $name            --lr_policy step      \
    --patch_size 128             --niter 400           --save_imgs True      --lr 1e-4      --dataroot $dataroot   \
    --batch_size 36          --print_freq 500      --calc_metrics True     --weight_decay 0.01 \
    --gpu_ids $device  -j 8   --lr_decay_iters 27 --block Convnext  --load_optimizers False  | tee $LOG 

    
