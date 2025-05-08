import sys
import torch
import torchvision

slice_layer_cnt = [
    2, 1, 1, 2, 4, 3, 3, 7
]

def main():
    batch_size = int(sys.argv[1])

    net = torchvision.models.mobilenet_v2(
        weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
    net = net.eval()
    x = torch.ones(batch_size, 3, 224, 224)
    torch.onnx.export(net, x, 'mobilenetv2.onnx',
        input_names=['input'],
        output_names=['output']
    )

    x = torch.ones(batch_size, 3, 224, 224)
    layers = []
    for name, layer in net.named_children():
        if name == 'classifier':
            layers.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
            layers.append(torch.nn.Flatten())
        for layer_1 in layer.children():
            layers.append(layer_1)
    
    layer_idx_begin = 0
    for slice_id, cnt in enumerate(slice_layer_cnt):
        layer_idx_end = layer_idx_begin + cnt
        slice = layers[layer_idx_begin:layer_idx_end]
        layer_idx_begin = layer_idx_end
        slice = torch.nn.Sequential(*slice).eval()
        print("slice", slice_id, slice)
        torch.onnx.export(slice, x, f'{slice_id}.onnx',
            input_names=['input'],
            output_names=['output']
        )
        x = slice(x)

    print(f"export done with {len(slice_layer_cnt)} slices")

if __name__ == '__main__':
    main()
