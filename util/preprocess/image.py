from PIL import Image
import numpy as np
import pathlib
img = 'util/preprocess/pytorch_logo.png'
dtype = np.float32

def save_img(out, size):
      pathlib.Path(out).parent.mkdir(exist_ok=True)
      resized_img = Image.open(img).convert(mode='RGB').resize(size)
      img_data = np.asarray(resized_img).astype(np.float32)
      img_data = np.transpose(img_data, (2, 0, 1))

      imagenet_mean = np.array([0.485, 0.456, 0.406])
      imagenet_stddev = np.array([0.229, 0.224, 0.225])
      norm_img_data = np.zeros(img_data.shape).astype(np.float32)

      for i in range(img_data.shape[0]):
            norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - imagenet_mean[i]) / imagenet_stddev[i]

      img_data = np.expand_dims(norm_img_data, axis=0)
      img_data.tofile(out)
      print(f'save img {img_data.shape} to {out}')
      
save_img('data/resnet/input-0.bin', (224, 224))
save_img('data/inception/input-0.bin', (299, 299))