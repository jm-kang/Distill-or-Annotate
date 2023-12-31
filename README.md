# Distill or Annotate? Cost-Efficient Fine-Tuning of Compact Models
This repository contains resources for the ACL 2023 paper [Distill or Annotate? Cost-Efficient Fine-Tuning of Compact Models](https://aclanthology.org/2023.acl-long.622.pdf).


![alt text](https://jm-kang.github.io/assets/publications/2023_dist_or_ann/strategies.png)


If you find our code, data or the paper useful, please cite the paper:
```
@inproceedings{kang-etal-2023-distill,
  title={Distill or Annotate? Cost-Efficient Fine-Tuning of Compact Models},
  author={Kang, Junmo and Xu, Wei and Ritter, Alan},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year={2023}
}
```

## Preparation

#### Installation
```
conda create --name <env> --file requirements.txt
```

#### Download Unlabeled Data
For running distillation, download [this unlabeled data](https://drive.google.com/file/d/1KaP6FDIZWgFWlT566IgXsJ6qAT0Z3FVJ/view?usp=sharing) for each task. Training/test sets are directly provided in each directory in this repository.


## Annotation Costs
Here are the estimates of annotation cost per label for each dataset. You are encouraged to calculate the number of train data based on your fixed budget, given the following estimates. For more details, please refer to the paper.

| WLP  | Stanceosaurus | FEVER | MultiPIT_id | MultiPIT_gen | NQ |
|---|---|---|---|---|---|
| $0.260 | $0.364 | $0.129 | $0.200 | $0.371 | $0.129 |


## Running Distillation Models

```
task=<task>
teacher_model=<teacher_model_path>
model=google/t5-v1_1-small
model_parallel_gpus=2
train_file=<unlabled_file>
dev_file=<dev_or_test_file>
train_batch=32
eval_batch=32
grad_accum=1
lr=3e-5
epochs=50
output_dir=<output_dir>
save_steps=5000
eval_steps=2000

time python ../run_t5.py \
  --distillation \
  --task $task \
  --model_parallel_gpus $model_parallel_gpus \
  --teacher_model_name_or_path $teacher_model \
  --model_name_or_path $model \
  --output_dir $output_dir \
  --do_eval \
  --validation_file $dev_file \
  --per_device_eval_batch_size $eval_batch \
  --predict_with_generate \
  --source_prefix "<task>: " \
  --source_column "source" \
  --target_column "target" \
  --do_train \
  --train_file $train_file \
  --per_device_train_batch_size $train_batch \
  --gradient_accumulation_steps $grad_accum \
  --num_train_epochs $epochs \
  --learning_rate $lr \
  --save_steps $save_steps \
  --evaluation_strategy "steps" \
  --eval_steps $eval_steps \
  --report_to wandb \
  --overwrite_output_dir
```

## Running Annotation Models

```
task=<task>
model=google/t5-v1_1-xxl
model_parallel_gpus=4
train_file=<train_data>
dev_file=<dev_or_test_data>
train_batch=32
eval_batch=32
grad_accum=1
lr=3e-5
epochs=20
output_dir=<output_dir>
save_steps=50000
eval_steps=2000

time python ../run_t5.py \
  --task $task \
  --model_parallel_gpus $model_parallel_gpus \
  --model_name_or_path $model \
  --output_dir $output_dir \
  --do_eval \
  --validation_file $dev_file \
  --per_device_eval_batch_size $eval_batch \
  --predict_with_generate \
  --source_prefix "<task>: " \
  --source_column "source" \
  --target_column "target" \
  --do_train \
  --train_file $train_file \
  --per_device_train_batch_size $train_batch \
  --gradient_accumulation_steps $grad_accum \
  --num_train_epochs $epochs \
  --learning_rate $lr \
  --save_steps $save_steps \
  --evaluation_strategy "steps" \
  --eval_steps $eval_steps \
  --report_to wandb \
  --overwrite_output_dir
```
