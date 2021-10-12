import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications import resnet50
import matplotlib.pyplot as plt
import numpy as np

# gpu config
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)

filename = 'cat.jpg'

original = load_img(filename, target_size = (224, 224))
print('PIL image size', original.size)

numpy_image = img_to_array(original)
plt.imshow(np.uint8(numpy_image))
print('numpy array size',numpy_image.shape)
image_batch = np.expand_dims(numpy_image, axis = 0)
print('image batch size', image_batch.shape)

processed_image = resnet50.preprocess_input(image_batch.copy())
resnet_model = resnet50.ResNet50(weights = 'imagenet')
predictions = resnet_model.predict(processed_image)
label = decode_predictions(predictions)
print(label)

plt.imshow(original)
plt.show()
