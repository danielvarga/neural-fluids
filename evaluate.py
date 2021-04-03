import sys
import numpy as np
import tensorflow as tf
from PIL import Image


from tensorflow import keras


def vis(state):
    rgb = state[:, :, :3]
    rgb = (np.clip(rgb, 0, 1) * 255).astype('uint8')
    return Image.fromarray(rgb) # .save(f'{FRAME_PATH}_{frame:03d}.png')

def concat(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


model_path, data_file = sys.argv[1:]

model = keras.models.load_model(model_path)

flow = np.load(open(data_file, "rb"))

entry_point = 20
state = flow[entry_point]
print(state.shape)

for i in range(len(flow) - entry_point):
    ground_truth = flow[entry_point + i]
    previous = flow[entry_point + i - 1]
    stateless = model.predict(previous[np.newaxis, :])[0]

    ground_truth_img = vis(ground_truth)
    state_img = vis(state)
    stateless_img = vis(stateless)
    concat(concat(ground_truth_img, stateless_img), state_img).save(f"compare_{i:03d}.png")

    state = model.predict(state[np.newaxis, :])[0]
