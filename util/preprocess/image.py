from PIL import Image
import numpy as np
import sys

img = sys.argv[1]
out = sys.argv[2]


resized_img = Image.open(img).resize((224,224))
img_data = np.asarray(resized_img).astype('float32')
img_data = np.transpose(img_data, (2, 0, 1))

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_stddev = np.array([0.229, 0.224, 0.225])
norm_img_data = np.zeros(img_data.shape).astype("float32")

for i in range(img_data.shape[0]):
      norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - imagenet_mean[i]) / imagenet_stddev[i]

img_data = np.expand_dims(norm_img_data, axis=0)
print(img_data.shape)
img_data.tofile(out)
