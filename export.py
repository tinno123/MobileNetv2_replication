import torch
from models.MobileNetV2 import MobileNetV2
from utils.train_tools import load_weights_by_shape

model = MobileNetV2(num_classes=1000)
load_weights_by_shape(model, 'weights/MobelNetV2_Weight.pth')

# model.load_state_dict(torch.load('weights/MobelNetV2_Weight.pth' , map_location='cpu'))
model.eval()

onnx_path = "./weights/imagenet.onnx"


input_shape = (1,3,224,224)
onnx_input = torch.rand(input_shape)
try:
    torch.onnx.export(
                model,
                onnx_input,  # 示例输入
                onnx_path,  # 输出路径
                export_params=True,  # 导出参数
                opset_version=11,  # ONNX算子版本
                do_constant_folding=True,  # 常量折叠优化
                input_names=['input'],
                output_names=['output']  # 输出名称
    )
    print(f"onnx 模型导出成功: {onnx_path}")
except Exception as e:
        print(e)