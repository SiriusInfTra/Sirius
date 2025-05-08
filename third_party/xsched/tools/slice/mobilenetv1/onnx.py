import torch
import torch.nn as nn

def conv_st(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

def conv_dw(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )

def conv_x5(in_channel, out_channel, blocks):
    layers = []
    for i in range(blocks):
        layers.append(conv_dw(in_channel, out_channel, 1))
    return nn.Sequential(*layers)

class MobleNetV1(nn.Module):
    def __init__(self, num_classes):
        super(MobleNetV1, self).__init__()
        self.conv1 = conv_st(3, 32, 2)
        self.conv_dw1 = conv_dw(32, 64, 1)
        self.conv_dw2 = conv_dw(64, 128, 2)
        self.conv_dw3 = conv_dw(128, 128, 1)
        self.conv_dw4 = conv_dw(128, 256, 2)
        self.conv_dw5 = conv_dw(256, 256, 1)
        self.conv_dw6 = conv_dw(256, 512, 2)
        self.conv_dw_x5 = conv_x5(512, 512, 5)
        self.conv_dw7 = conv_dw(512, 1024, 2)
        self.conv_dw8 = conv_dw(1024, 1024, 1)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, num_classes)
        self.softmax = nn.Softmax(dim=1)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_dw1(x)
        x = self.conv_dw2(x)
        x = self.conv_dw3(x)
        x = self.conv_dw4(x)
        x = self.conv_dw5(x)
        x = self.conv_dw6(x)
        x = self.conv_dw_x5(x)
        x = self.conv_dw7(x)
        x = self.conv_dw8(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x


net = MobleNetV1(1024)
x = torch.zeros(1, 3, 224, 224)
for name, layer in net.named_children():
    torch.onnx.export(layer, x, f'{name}.onnx')
    x = layer(x)
    print(name, 'output shape:', x.shape)


net.eval()
x = torch.zeros(1, 3, 224, 224)
torch.onnx.export(net, x, 'mobilenet.onnx',
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)