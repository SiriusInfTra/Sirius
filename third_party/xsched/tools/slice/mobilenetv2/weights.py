# from https://github.com/wang-xinyu/pytorchx/blob/master/mobilenet/inference.py
# from https://github.com/wang-xinyu/pytorchx/blob/master/mobilenet/mobilenetv2.py

import os
import struct
import torch
import torchvision

def main():
    net = torchvision.models.mobilenet_v2(pretrained=True)
    net = net.eval()
    tmp = torch.ones(1, 3, 224, 224)
    out = net(tmp)
    print('mobilenet out:', out.shape)

    # save wts
    f = open("mobilenetv2.wts", 'w')
    f.write("{}\n".format(len(net.state_dict().keys())))
    for k,v in net.state_dict().items():
        print('key: ', k)
        print('value: ', v.shape)
        vr = v.reshape(-1).cpu().numpy()
        f.write("{} {}".format(k, len(vr)))
        for vv in vr:
            f.write(" ")
            f.write(struct.pack(">f", float(vv)).hex())
        f.write("\n")

if __name__ == '__main__':
    main()
