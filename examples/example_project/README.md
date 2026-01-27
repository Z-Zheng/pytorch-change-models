
### Example project: benchmark DINOv3-based ChangeStar(1x256) model on BRIGHT dataset


```bash
# remove --use_wandb and --project if you don't have wandb account
config_path='configs/dinov3_cstar_vitl_1x256.py'
model_dir='logs/bright_dinov3_cstar_vitl_1x256'
torchrun --nnodes=1 --nproc_per_node=2 --master_port $RANDOM -m torchange.training.bisup_train_bright \
  --config_path=${config_path} \
  --model_dir=${model_dir} \
  --mixed_precision='bf16' \
  --use_wandb \
  --project 'torchange_example_project_BRIGHT_bench' \
  --eval_epoch_interval 5 \
  data.train.params.batch_size 8 \
  data.train.params.num_workers 4
```