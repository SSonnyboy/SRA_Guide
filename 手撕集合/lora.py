import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # 原始权重矩阵 (冻结，不更新)
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim), requires_grad=False)
        
        # LoRA 适配器
        self.lora_A = nn.Parameter(torch.randn(rank, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        
        # 缩放因子
        self.scaling = alpha / rank
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 原始权重计算
        output = F.linear(x, self.weight)
        
        # LoRA 适配器计算
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B)
        
        return output + self.scaling * lora_output
    
def apply_lora_to_linear_layer(module, rank=8, alpha=16):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            # 替换线性层为 LoRALayer
            lora_layer = LoRALayer(
                child.in_features, 
                child.out_features, 
                rank=rank, 
                alpha=alpha
            )
            # 复制原始权重
            lora_layer.weight.data = child.weight.data.clone()
            setattr(module, name, lora_layer)
        else:
            # 递归应用
            apply_lora_to_linear_layer(child, rank, alpha)

# 示例：应用于预训练模型
model = ...  # 你的预训练模型
apply_lora_to_linear_layer(model, rank=4, alpha=8)

# 只训练 LoRA 参数
for name, param in model.named_parameters():
    if 'lora_A' in name or 'lora_B' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

def merge_lora_weights(model):
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.weight.data += module.scaling * module.lora_B @ module.lora_A

# from peft import LoraConfig, get_peft_model

# config = LoraConfig(
#     r=8,
#     lora_alpha=16,
#     target_modules=["query", "value"],
#     lora_dropout=0.1,
#     bias="none",
#     init_lora_weights="gaussian"  # 或 "pissa" 表示更高级的初始化
# )

# model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
# model = get_peft_model(model, config)
