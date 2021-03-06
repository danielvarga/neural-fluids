import numpy as np
from PIL import Image

from fluid import Fluid

FRAME_PATH = 'placeholder'

N = 128
RESOLUTION = N, N
VISCOSITY = 10 ** -3
DURATION = 200

INFLOW_PADDING = 20
INFLOW_RADIUS = 8
INFLOW_VELOCITY = 2

channels = 'r', 'g', 'b'

def circle(theta):
    return np.asarray((np.cos(theta), np.sin(theta)))


def save(fluid):
    a = np.stack([fluid.quantities['r'], fluid.quantities['g'], fluid.quantities['b'], fluid.velocity_field[:, 0], fluid.velocity_field[:, 1]], axis=1)
    a = a.reshape((*RESOLUTION, len(channels) + 2))
    return a.astype(np.float32)


def save_div_press(fluid):
    divergence = sum(fluid.gradient[d].dot(fluid.velocity_field[..., d]) for d in range(fluid.dimensions))
    pressure = fluid.pressure_solver(divergence)
    divergence = divergence.reshape(*RESOLUTION)
    pressure = pressure.reshape(*RESOLUTION)

    return np.stack([divergence, pressure])


def main():
    center = np.floor_divide(RESOLUTION, 2)
    r = np.min(center) - INFLOW_PADDING
    directions = tuple(-circle(p * np.pi * 2 / 3) for p in range(3))
    points = tuple(r * circle(p * np.pi * 2 / 3) + center for p in range(3))

    fluid = Fluid(RESOLUTION, VISCOSITY, channels, solver_type=("jacobi", 100))

    inflow_dye_field = np.zeros((fluid.size, len(channels)))
    inflow_velocity_field = np.zeros_like(fluid.velocity_field)
    for i, p in enumerate(points):
        distance = np.linalg.norm(fluid.indices - p, axis=1)
        mask = distance <= INFLOW_RADIUS

        for d in range(2):
            inflow_velocity_field[..., d][mask] = directions[i][d] * INFLOW_VELOCITY

        inflow_dye_field[..., i][mask] = 1

    print(inflow_velocity_field.shape, inflow_dye_field.shape)

    print(fluid.velocity_field.shape, fluid.quantities['r'].shape)

    inflow_velocity_field += np.random.normal(scale=0.05, size=inflow_velocity_field.shape)

    fluid.velocity_field += inflow_velocity_field

    for i, k in enumerate(channels):
        fluid.quantities[k] += inflow_dye_field[..., i]

    states = []
    div_presses = []
    for frame in range(DURATION):
        # print(f'Computing frame {frame}.')

        fluid.project()

        rgb = np.dstack(tuple(fluid.quantities[c] for c in channels))

        rgb = rgb.reshape((*RESOLUTION, 3))
        rgb = (np.clip(rgb, 0, 1) * 255).astype('uint8')
        Image.fromarray(rgb).save(f'{FRAME_PATH}_{frame:05d}.png')

        fluid.advect_diffuse()
        states.append(save(fluid))
        div_presses.append(save_div_press(fluid))

    states = np.array(states)
    div_presses = np.array(div_presses)
    print("div_presses.shape", div_presses.shape)

    np.save(open("corpus.npy", "wb"), states)
    np.save(open("divergence-to-pressure.npy", "wb"), div_presses)


main()
