from pathlib import Path

import torch
import torchvision.models as models

# 1. 加载预训练模型 (推荐方式)
# 使用 IMAGENET1K_V2 权重，这是默认的，也是精度最高的
model = models.mobilenet_v2(weights='IMAGENET1K_V2')
torch.save(model.state_dict(), Path("./") / "MobelNetV2.pth")