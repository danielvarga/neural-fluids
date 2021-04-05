# python evaluate_pressure.py model-pressure.mdl divergence-to-pressure.npy 

import sys
import numpy as np
import tensorflow as tf
from PIL import Image


from tensorflow import keras


def vis(p):
    c = 1
    c_uint = ((np.clip(p / c, -1, 1) / 2 + 1) * 255).astype('uint8')[:, :, 0]
    return Image.fromarray(c_uint) # .save(f'{FRAME_PATH}_{frame:03d}.png')

def concat(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


model_path, data_file = sys.argv[1:]

model = keras.models.load_model(model_path)

div_press = np.load(open(data_file, "rb"))
x = div_press[:, 0, :, :, np.newaxis]
y = div_press[:, 1, :, :, np.newaxis]

entry_point = 20
for i in range(len(x) - entry_point):
    div = x[entry_point + i]
    ground_truth_pressure = y[entry_point + i]
    predicted_pressure = model.predict(div[np.newaxis, :])[0]

    ground_truth_img = vis(ground_truth_pressure)
    predicted_img = vis(predicted_pressure)
    concat(ground_truth_img, predicted_img).save(f"predicted-pressure-{i:03d}.png")
