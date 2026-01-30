# Copyright (c) Zhuo Zheng and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALayer:
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class LoraLinear(nn.Linear, LoRALayer):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool,
            r: int,
            lora_alpha: int,
            lora_dropout: float = 0.,
            merge_weights: bool = False,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias)
        LoRALayer.__init__(self, r, lora_alpha, lora_dropout, merge_weights)

        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = F.linear(x, self.weight, bias=self.bias)
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            return result
        else:
            return F.linear(x, self.weight, bias=self.bias)

    @classmethod
    def convert_lora_linear(cls, module, r, lora_alpha, lora_dropout=0., merge_weights=False):
        module_output = module
        if isinstance(module, nn.Linear):
            module_output = LoraLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                merge_weights=merge_weights,
            )
            with torch.no_grad():
                module_output.weight = module.weight
                if module.bias is not None:
                    module_output.bias = module.bias

            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(
                name, cls.convert_lora_linear(child, r, lora_alpha, lora_dropout, merge_weights)
            )
        del module
        return module_output

    def extra_repr(self) -> str:
        return f"{nn.Linear.extra_repr(self)}, rank={self.r}"


def lora_on_attention(vit, **kwargs):
    for block in vit.blocks:
        block.attn = LoraLinear.convert_lora_linear(block.attn, **kwargs)

    return vit
