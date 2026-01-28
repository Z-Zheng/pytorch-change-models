# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import ever as er
import fire
from huggingface_hub import ModelCard, ModelCardData
from huggingface_hub.utils import is_jsonable

DEFAULT_MODEL_CARD = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

- Model class: {{ model_name }}
- Source: {{ model_source }}
- Codebase: {{ library_name }}
"""


def _remove_non_json_serializable(cfg):
    rm = []
    for k, v in cfg.items():
        if isinstance(v, dict):
            _remove_non_json_serializable(v)
        elif not is_jsonable(v):
            rm.append(k)

    for k in rm:
        er.info(f'{k}={cfg[k]} is not JSON serializable, removed from config.')
        cfg.pop(k)


@er.dist.main_process_only
def model_dir_to_hub(model_dir, repo_id, private=False, checkpoint_name=None, model_card: ModelCard = None):
    er.registry.register_modules()
    model, _ = er.infer_tool.build_from_model_dir(model_dir=model_dir, checkpoint_name=checkpoint_name)

    full_name = f"{model.__module__}.{model.__class__.__name__}"
    model._hub_mixin_info.model_card_data.library_name = 'torchange'
    model._hub_mixin_info.model_card_data.model_name = model.__class__.__name__
    model._hub_mixin_info.model_card_data.model_source = full_name
    model._hub_mixin_info.model_card_template = DEFAULT_MODEL_CARD

    model: er.ERModule
    if model._hub_mixin_config is None:
        cfg = model.config.to_dict()
        _remove_non_json_serializable(cfg)
        model._hub_mixin_config = {
            list(model._hub_mixin_init_parameters)[1]: cfg
        }

    model.push_to_hub(repo_id=repo_id, private=private)

    if model_card is not None:
        model_card.data.library_name = 'torchange'
        model_card.data.model_name = model.__class__.__name__
        model_card.data.model_source = full_name
        model_card.push_to_hub(repo_id, repo_type='model')


if __name__ == '__main__':
    fire.Fire({
        'model_dir_to_hub': model_dir_to_hub
    })
