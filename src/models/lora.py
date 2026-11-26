from torch import nn
import torch
from peft import LoraConfig

class LoRAFFNWrapper(nn.Module):
    def __init__(self, original_ffn, rank, alpha, hidden_dim):
        super().__init__()
        self.block = original_ffn
        # 整体挂载的 LoRA 路径：A (d -> 128), B (128 -> d)
        self.lora_A = nn.Linear(hidden_dim, rank, bias=False)
        self.lora_B = nn.Linear(rank, hidden_dim, bias=False)
        self.scaling = alpha / rank

        # 高斯初始化 lora_A，标准差为 1/r
        nn.init.normal_(self.lora_A.weight, std=1.0 / rank)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # 原始 FFN 结果 + LoRA 增量
        return self.block(x) + self.lora_B(self.lora_A(x)) * self.scaling


def initialize_dit_ffn_big_lora(dit, lora_rank, lora_alpha):
    dit.requires_grad_(False)
    dit.train()

    ### step1: add Global LORA (rank 128) to FFN
    layers_to_replace = []
    for name, module in dit.named_modules():
        if "dit_blocks" in name and (name.endswith(".ff") or name.endswith(".ff_context")):
            layers_to_replace.append((name, module))

    for name, module in layers_to_replace:
        first_linear = next(m for m in module.modules() if isinstance(m, torch.nn.Linear))
        current_hidden_size = first_linear.in_features

        name_parts = name.split(".")
        child_name = name_parts[-1]
        parent_name = ".".join(name_parts[:-1])

        if parent_name:
            parent_module = dit.get_submodule(parent_name)
        else:
            parent_module = dit

        # 获取原始模块的 device 和 dtype
        param = next(module.parameters())
        device, dtype = param.device, param.dtype

        # 创建并替换
        new_ffn = LoRAFFNWrapper(
            module,
            rank=128,
            alpha=128,
            hidden_dim=current_hidden_size
        ).to(device=device, dtype=dtype)
        setattr(parent_module, child_name, new_ffn)
        new_ffn.lora_A.weight.requires_grad = True
        new_ffn.lora_B.weight.requires_grad = True
        print(f"✅ Applied Global LoRA (Rank 128) to FFN: {name} (Dim: {current_hidden_size})")

    ### step2: add Lora (rank 64) to other
    def get_specific_layer_names(model):
        target_types = (
            torch.nn.Linear,
            torch.nn.Embedding,
            torch.nn.Conv2d,
        )

        excluded_names = (
            "ff",
            "time_embed",
            "r_embedder",
            "context_embedder",
            "x_embedder",
            "norm_out",
            "proj_out",
            "norm1.linear",
            "norm1_context.linear",
        )

        return sorted(
            name
            for name, module in model.named_modules()
            if isinstance(module, target_types)
            and not any(key in name for key in excluded_names)
        )
    lora_target_modules = get_specific_layer_names(dit)

    lora_conf_dit = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, init_lora_weights="gaussian",
                               target_modules=lora_target_modules)

    dit.add_adapter(lora_conf_dit, adapter_name="default_dit")
    dit.enable_adapters()

    return dit, lora_target_modules
