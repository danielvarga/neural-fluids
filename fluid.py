import numpy as np
import scipy.sparse as sp
from scipy.ndimage import map_coordinates
from scipy.sparse.linalg import factorized

import operators as ops


def jacobi_solver(d, lplusu, b, iterations):
    x = np.ones(d.shape)
    for i in range(iterations):
        x = (b - lplusu.dot(x)) / d
    return x


class Fluid:
    # solver_type can be one of "dense", "sparse" or ("jacobi", iteration_count).
    def __init__(self, shape, viscosity, quantities, solver_type="sparse"):
        self.iteration = 0
        self.shape = shape
        # Defining these here keeps the code somewhat more readable vs. computing them every time they're needed.
        self.size = np.product(shape)
        self.dimensions = len(shape)

        # Variable viscosity, both in time and in space, is easy to set up; but it conflicts with the use of
        # SciPy's factorized function because the diffusion matrix must be recalculated every frame.
        # In order to keep the simulation speedy I use fixed viscosity.
        self.viscosity = viscosity

        # By dynamically creating advected-diffused quantities as needed prototyping becomes much easier.
        self.quantities = {}
        for q in quantities:
            self.quantities[q] = np.zeros(self.size)

        self.velocity_field = np.zeros((self.size, self.dimensions))
        # The reshaping here corresponds to a partial flattening so that self.indices
        # has the same shape as self.velocity_field.
        # This makes calculating the advection map as simple as a single vectorized subtraction each frame.
        self.indices = np.dstack(np.indices(self.shape)).reshape(self.size, self.dimensions)
        print("indices", self.indices.shape)

        self.gradient = ops.matrices(shape, ops.differences(1, (1,) * self.dimensions), False)
        g = self.gradient[0]
        print("gradient", len(self.gradient), g.shape, type(g), g.data.shape, g.row.shape)

        # Both viscosity and pressure equations are just Poisson equations similar to the steady state heat equation.
        self.laplacian = ops.matrices(shape, ops.differences(1, (2,) * self.dimensions), True)
        print("laplacian", self.laplacian.shape, type(self.laplacian))

        n = shape[0]
        '''
        lap_inv = lap_inv.reshape((n, n, n, n))
        import matplotlib.pyplot as plt
        plt.imshow(lap_inv[n // 2, n // 2, :, :])
        plt.show()
        '''

        '''
        L = splu_laplacian.L.toarray().reshape((n, n, n, n))
        print("L", L.shape)
        import matplotlib.pyplot as plt
        plt.imshow(L[n // 2, n // 2, :, :])
        plt.show()
        exit()
        '''

        if solver_type == "sparse":
            from scipy.sparse.linalg import splu
            splu_laplacian = splu(self.laplacian)
            self.pressure_solver = splu_laplacian.solve

            # Making sure I use the sparse version of the identity function here so I don't cast to a dense matrix.
            self.viscosity_solver = splu(sp.identity(self.size) - self.laplacian * viscosity).solve
        elif solver_type == "dense":
            print("the dense codepath is very slow, it is here strictly for pedagogical reasons.")
            print("solving dense pressure equation...")
            lap_inv = np.linalg.inv(self.laplacian.toarray())
            self.pressure_solver = lambda x: x.dot(lap_inv)
            print("done")

            print("solving dense viscosity equation...")
            visc_inv = np.linalg.inv((sp.identity(self.size) - laplacian * viscosity).toarray())
            self.viscosity_solver = lambda x: x.dot(visc_inv)
            print("done")
        elif solver_type[0] == "jacobi":
            import scipy
            self.laplacian_diag = self.laplacian.diagonal()
            self.laplacian_nondiag = self.laplacian - scipy.sparse.spdiags(self.laplacian_diag, 0, self.laplacian_diag.size, self.laplacian_diag.size)
            iterations = solver_type[1]

            self.viscosity_diag = 1 - self.laplacian_diag * viscosity
            self.viscosity_nondiag = - self.laplacian_nondiag * viscosity

            self.pressure_solver = lambda b: jacobi_solver(self.laplacian_diag, self.laplacian_nondiag, b, iterations)
            self.viscosity_solver = lambda b: jacobi_solver(self.viscosity_diag, self.viscosity_nondiag, b, iterations)

        else:
            assert False, f"unknown solver_type {solver_type}"

    def advect_diffuse(self):
        # Advection is computed backwards in time as described in Jos Stam's Stable Fluids whitepaper.
        advection_map = np.moveaxis(self.indices - self.velocity_field, -1, 0)

        def kernel(field):
            # Credit to Philip Zucker for pointing out the aptness of map_coordinates here.
            # Initially I was using SciPy's griddata function.
            # While both of these functions do essentially the same thing, griddata is much slower.
            advected = map_coordinates(field.reshape(self.shape), advection_map, order=2).flatten()
            return self.viscosity_solver(advected) if self.viscosity > 0 else advected

        # Apply viscosity and advection to each axis of the velocity field and each user-defined quantity.
        for d in range(self.dimensions):
            self.velocity_field[..., d] = kernel(self.velocity_field[..., d])

        for k, q in self.quantities.items():
            self.quantities[k] = kernel(q)

    def project(self):
        self.iteration += 1
        # Pressure is calculated from divergence which is in turn calculated from the gradient of the velocity field.
        divergence = sum(self.gradient[d].dot(self.velocity_field[..., d]) for d in range(self.dimensions))
        pressure = self.pressure_solver(divergence)

        visualize_pressure = False
        if visualize_pressure and self.iteration == 10:
            import matplotlib.pyplot as plt
            plt.imshow(divergence.reshape(self.shape))
            plt.axis("off")
            plt.title("divergence")
            plt.savefig("divergence.png")
            for iterations_here in [1, 2, 5, 10, 20, 50, 100, 200, 1000]:
                pressure = jacobi_solver(self.laplacian_diag, self.laplacian_nondiag, divergence, iterations_here)
                pressure -= pressure.mean()
                print("jacobi", iterations_here, "pressure min/max", pressure.min(), pressure.max())
                plt.imshow(pressure.reshape(self.shape), vmin=-1, vmax=1)
                plt.title(f"pressure / jacobi solver, {iterations_here} iterations")
                plt.axis("off")
                plt.savefig(f"pressure-it-{iterations_here:04d}.png")

            print("starting sparse")
            from scipy.sparse.linalg import splu
            splu_laplacian = splu(self.laplacian)
            sparse_pressure_solver = splu_laplacian.solve
            sparse_pressure = sparse_pressure_solver(divergence)
            sparse_pressure -= sparse_pressure.mean()
            print("sparse min/max", sparse_pressure.min(), sparse_pressure.max())
            plt.imshow(sparse_pressure.reshape(self.shape), vmin=-1, vmax=1)
            plt.axis("off")
            plt.title("pressure / sparse solver")
            plt.savefig("pressure-sparse.png")

            print("starting dense")
            lap_inv = np.linalg.inv(self.laplacian.toarray())
            dense_pressure_solver = lambda x: x.dot(lap_inv)
            dense_pressure = dense_pressure_solver(divergence)
            dense_pressure -= dense_pressure.mean()
            print("dense min/max", dense_pressure.min(), dense_pressure.max())
            plt.imshow(dense_pressure.reshape(self.shape), vmin=-1, vmax=1)
            plt.axis("off")
            plt.title("pressure / dense solver")
            plt.savefig("pressure-dense.png")
            exit()

        for d in range(self.dimensions):
            self.velocity_field[..., d] -= self.gradient[d].dot(pressure)
