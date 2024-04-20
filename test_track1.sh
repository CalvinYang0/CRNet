#!/bin/bash
echo "Start to test the model...."
device="0"


dataroot=""  # including 'Train' and 'NTIRE_Val' floders
name=""

python test.py \
    --dataset_name bracketire   --model cat       --name $name          --dataroot $dataroot  \
    --load_iter 500            --save_imgs True      --calc_metrics False  --gpu_id $device  -j 8   --block Convnext



