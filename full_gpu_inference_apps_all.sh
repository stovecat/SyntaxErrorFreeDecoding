#! /bin/bash
TOKENIZERS_PARALLELISM=false

name_path="./apps/train/finetuned/Salesforce_codegen-2B-mono/10-04-2022__10:29:30/checkpoint-"
ckpt=2000
name=${name_path}${ckpt}

n_sample=30
dataset_name="apps"
dataset_type="train"
max_description_tokens=500
max_new_tokens=512
max_length=$(($max_description_tokens + $max_new_tokens))
p=0.95
T=0.8

timestamp=`date +%Y-%m-%d_%H-%M-%S`
gpu_start_idx=0
n_gpus=16
count=0
while [ $count -lt $n_gpus ]
do
    start_index=`expr $((5000 / $n_gpus)) \* $count`
    start_index=${start_index%.*}
    end_index=`expr $((5000 / $n_gpus)) \* $(($count+1))`
    end_index=${end_index%.*}
    if [ $count -eq 5 ] || [ $count -eq 6 ] || [ $count -eq 8 ]
    then
        if [ $count -eq $(($n_gpus-1)) ]
        then
            end_index=5000
            CUDA_VISIBLE_DEVICES=$(($count+$gpu_start_idx)) python single_gpu_inference.py --start_idx $start_index --end_idx $end_index --dataset_name $dataset_name --time $timestamp --num_return_sequences $n_sample --max_length $max_length --max_new_tokens $max_new_tokens --dataset_type $dataset_type --decoding nucleus --T $T --model_name $name --apps_max_description_tokens $max_description_tokens --max_length $max_length --syntax_error_free  2>&1| tee logs/${timestamp}_${name//\//\_}_nucleus_p\=${p}_T\=${T}_${n_sample}_${start_index}_${end_index}.log
        else
            CUDA_VISIBLE_DEVICES=$(($count+$gpu_start_idx)) python single_gpu_inference.py --start_idx $start_index --end_idx $end_index --dataset_name $dataset_name --time $timestamp --num_return_sequences $n_sample --max_length $max_length --max_new_tokens $max_new_tokens --dataset_type $dataset_type --decoding nucleus --T $T --model_name $name --apps_max_description_tokens $max_description_tokens --max_length $max_length --syntax_error_free  2>&1| tee logs/${timestamp}_${name//\//\_}_nucleus_p\=${p}_T\=${T}_${n_sample}_${start_index}_${end_index}.log &
        fi        
    fi
    count=$(($count+1))
done
