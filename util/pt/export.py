import torch
from torchvision import models
import tempfile

batch_size = 1
model_store = "server/models"
tmp_dir = tempfile.gettempdir()

resnet152 = models.resnet152(weights=models.ResNet152_Weights.DEFAULT).eval().cuda()
scripted_model = torch.jit.script(resnet152)
scripted_model.save('resnet152.pt')