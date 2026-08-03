## 🚀 xView2 project: Swin-based ChangeOS for building damage assessment

Trains [**ChangeOS**](https://www.sciencedirect.com/science/article/abs/pii/S0034425721003564) with a
Swin-T backbone on [xView2](https://xview2.org/), scored with the official mixed metric
(`0.3 * localization F1 + 0.7 * harmonic_mean(damage F1s)`).

Unlike `examples/example_project` (BRIGHT), this project uses the dataset-agnostic entry point
`torchange.training.bisup_train` — the evaluator is declared in `config.train.callbacks` rather than
hardcoded in the training script.

### 📊 Performance Summary

Trained on `train`+`tier3`, model selected on `test`, final numbers reported on `hold`.

| arch         | backbone | Overall F1 | Localization F1 | Damage F1 | no-damage | minor | major | destroyed | Weights |
|:-------------|:---------|:----------:|:---------------:|:---------:|:----------|:------|:------|:----------|:--------|
| **ChangeOS** | Swin-T   |   76.91    |      85.03      |   73.43   | 89.39     | 58.03 | 72.75 | 81.25     | [🤗link](https://huggingface.co/EVER-Z/torchange_example_changeos_swint_on_xview2_best42k) |

`configs/swint_cos.py` as committed: 60k iters, effective batch 16, poly LR from 6e-5, bf16;
3 h 16 min on 2 × A100-40GB.

Selection picked step 42,619 of 60,000. `test/final_f1` over training:

| step | 10,654 | 21,309 | 31,964 | **42,619** | 53,274 | 60,000 |
|:---|:---|:---|:---|:---|:---|:---|
| final F1 | 75.36 | 75.99 | 76.40 | **76.68** | 76.05 | 75.89 |

The last two evaluations are worse than the best, so the reported model is `model-best.pth`, *not*
`checkpoint-60000.pth` — which is the point of keeping the two separate.

---

### 1. Data preparation (run once, before training)

Two things happen here, and both must happen **outside** the DDP training job:

1. **Download.** `EVER-Z/torchange_xView2` is ~24 GB (train 2799, tier3 3378, test 933, hold 933).
   Set `HF_HOME` first if your home directory is small or quota-limited.
2. **The valid-patch index.** `HFxView2.build_index()` decodes every `t2_mask` and scans
   6177 × 9 tiles to drop tiles with no damage label, caching the result as
   `HFxView2_train_tier3_valid_indices_p512_s256.npy` in the **current working directory**. It has no
   rank guard, so under `torchrun` every rank would redo the whole scan and race on the write.

Driving it from the config guarantees the cache filename matches what training looks for:

```bash
cd examples/xview2_project

python -c "
import ever as er
er.registry.register_all()
import torchange  # noqa: F401 -- registers HFxView2
cfg = er.config.import_config('configs/swint_cos.py')
ds = er.registry.DATASET[cfg.data.train['type']](cfg.data.train['params'])
print(len(ds), 'valid patches')
"
```

Construct the **dataset**, not the dataloader: `er.builder.make_dataloader` would go on to build a
`StepDistributedSampler`, which raises `Default process group has not been initialized` outside a
`torchrun` context. Downloading and indexing only need the dataset.

For `train`+`tier3` at 512/256 this keeps **34,081** of 55,593 candidate patches.

### 2. Training

```bash
cd examples/xview2_project    # cwd matters: er.registry.register_all() and `from configs import ...`

# remove --use_wandb and --project if you don't have a wandb account
torchrun --nnodes=1 --nproc_per_node=2 --master_port $RANDOM \
  -m torchange.training.bisup_train \
  --config_path=configs/swint_cos.py \
  --model_dir=logs/xview2_swint_cos \
  --mixed_precision='bf16' \
  --use_wandb \
  --project 'torchange_xview2_bench'
```

Trailing bare `key value` pairs override the config, e.g.
`data.train.params.batch_size 8`, `train.callbacks.0.params.epoch_interval 5`.

At batch 8 × 2 GPUs the 34,081 patches give 2,130 steps/epoch, so 60k iters ≈ 28 epochs.
`train.num_iters` is the main knob — keep `learning_rate.params.max_iters` equal to it.

`resume_from_last` defaults to `True` and the config writes a checkpoint every 2 epochs, so an
interrupted run only needs the same command re-submitted with the same `--model_dir`. The eval
callback restores its score history from `test_tracked_scores.csv` on restart, so a resume cannot
replace `model-best.pth` with a worse model or truncate the CSV.

> **On Slurm**, request the GPUs on a *single node* (e.g. `-N 1 -G 2`). Given only `-G 2`, Slurm may
> allocate one GPU on each of two nodes, and `torchrun --nnodes=1 --nproc_per_node=2` then fails on
> rank 1 with `CUDA error: invalid device ordinal`. Check with
> `sacct -j <id> --format=AllocTRES%60` — you want `gres/gpu=2,node=1`.

### 3. Final evaluation on `hold`

The protocol is **train on `train`+`tier3` → select on `test` → report on `hold`**. The training-time
callback evaluates `test` every 5 epochs and saves `model-best.pth` at the best `test/final_f1`;
`hold` is touched exactly once, afterwards.

`hold` deliberately is **not** a second training callback. `_xView2StandardEval.func()` writes
`model-best.pth` whenever *its own* split improves, and both callbacks would write the same
filename — a `hold` callback would silently overwrite the `test`-selected checkpoint. So it is a
separate step:

```bash
cd examples/xview2_project
torchrun --nnodes=1 --nproc_per_node=2 --master_port $RANDOM eval_hold.py \
  --model_dir logs/xview2_swint_cos
```

With no `--checkpoint_name`, `er.infer_tool.build_from_model_dir` loads `model-best.pth` — exactly
the `test`-selected weights. Results go to `hold_scores.csv` in the model dir. Use
`--nproc_per_node=1` for a single GPU; torchrun is required either way because
`er.builder.make_dataloader` constructs a `StepDistributedSampler`.

### 4. Export

```bash
python -m torchange.utils.push_to_hub model_dir_to_hub \
  --model_dir logs/xview2_swint_cos \
  --repo_id <your hf username>/<repo name> \
  --private
```

Needs the `hub` extra (`fire` + `huggingface-hub`). Note that `model_dir_to_hub` rebuilds the model
from `config.pkl`, which — unlike the config `.py` — does not re-run the algorithm import, so import
`torchange.models.changeos` yourself if you call it from Python rather than the CLI.

### ⚠️ Notes

- **Eval `batch_size` must stay 1.** `torchange/metrics/xview2.py` applies the single-map constraint
  as `dam_pred = loc_pred * dam_pred` with shapes `(B,1,H,W) * (B,H,W)`, which only broadcasts
  correctly at `B=1`. `HFxView2StandardEval` hardcodes it.
- `ChangeOS.set_default_config()` is already xView2-shaped (`loc_head.num_classes=1`,
  `dam_head.num_classes=5`), so `configs/swint_cos.py` only sets the decoder widths.
- `HFxView2` has no `ignore_t2_bg` option (only the file-path `xView2` class does), so damage class 0
  is trained as background — which is what the eval-time single-map constraint assumes.
- `torchange/models/` is not auto-imported, so the config imports `ChangeOS` itself at the top.
  Evaluators, datasets and modules are auto-imported and can be named by string in the config.

### 📚 References

```bibtex
@software{zheng2024torchange,
  author = {Zheng, Zhuo},
  title = {torchange: A Unified Change Representation Learning Benchmark Library},
  url = {https://github.com/Z-Zheng/pytorch-change-models},
  year = {2024}
}

@article{zheng2021changeos,
  title={Building damage assessment for rapid disaster response with a deep object-based semantic change detection framework: From natural disasters to man-made disasters},
  author={Zheng, Zhuo and Zhong, Yanfei and Wang, Junjue and Ma, Ailong and Zhang, Liangpei},
  journal={Remote Sensing of Environment},
  volume={265},
  pages={112636},
  year={2021},
  publisher={Elsevier}
}

@article{gupta2019xbd,
  title={xBD: A dataset for assessing building damage from satellite imagery},
  author={Gupta, Ritwik and Hosfelt, Richard and Sajeev, Sandra and Patel, Nirav and Goodman, Bryce and Doshi, Jigar and Heim, Eric and Choset, Howie and Gaston, Matthew},
  journal={arXiv preprint arXiv:1911.09296},
  year={2019}
}

@inproceedings{liu2021swin,
  title={Swin transformer: Hierarchical vision transformer using shifted windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and Hu, Han and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Guo, Baining},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={10012--10022},
  year={2021}
}
```
