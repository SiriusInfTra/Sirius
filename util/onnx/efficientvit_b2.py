import os
import subprocess
import torch
import timm



platform = 'v100'

batch_size = 1
# swin = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True).eval()

# vit_s_16 = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT).eval()
# traced_model = torch.jit.script(vit_b_16, torch.rand(batch_size, 3, 224, 224))
# vit_s_16 = ViTForImageClassification.from_pretrained('WinKawaks/vit-small-patch16-224')

model_name = 'efficientvit_b2'
# swin_t = models.swin_t(weights=models.Swin_T_Weights.DEFAULT).eval()
model = timm.create_model('efficientvit_b2.r224_in1k', pretrained=True)


torch.onnx.export(model, torch.rand(batch_size, 3, 224, 224), 
                  f"{model_name}.onnx",verbose=False,
                  input_names=["input0"], output_names=["output0"], 
                  export_params=True)


if 'COLSYS_TENSORRT_HOME' in os.environ:
    tensorrt_home = os.environ['COLSYS_TENSORRT_HOME']
    amd64_home = f'{tensorrt_home}/targets/x86_64-linux-gnu'
    envs = os.environ.copy()
    if 'LD_LIBRARY_PATH' in envs:
        envs['LD_LIBRARY_PATH'] = f'{amd64_home}/lib:{envs["LD_LIBRARY_PATH"]}'
    else:
        envs['LD_LIBRARY_PATH'] = f'{amd64_home}/lib'
    subprocess.run([f'{amd64_home}/bin/trtexec', f'--onnx={model_name}.onnx', f'--saveEngine={model_name}.plan'], 
        env=envs)