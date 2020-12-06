import torch
import torchvision

model = torchvision.models.resnet18()

# Smaller example
example = torch.randn(1, 3, 64, 64)
script_model = torch.jit.trace(model, example)

script_model.save("model.ptc")
