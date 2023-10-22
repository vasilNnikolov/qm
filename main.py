import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def get_force(A: float, N: int):
    return np.random.normal(0, A, (N, 2))


N_particles = 1000
N_iterations = 10000

M = 1
A = 1
dt = 0.01


def propagate_particle(x):
    pos = np.array([0.0, 0.0])
    vel = np.array([0.0, 0.0])

    Fs = get_force(A, N_iterations)
    positions = np.zeros((N_iterations, 2))

    for i in range(N_iterations):
        vel += Fs[i] / M
        pos += vel
        positions[i, :] = pos

    return positions


def main_no_multi():
    # contains the distance from the origin of a given particle on a given iteration
    positions = np.zeros((N_particles, N_iterations, 2))
    for particle_index in tqdm(range(N_particles)):
        positions[particle_index] = propagate_particle()

    # plot results
    N_plots = 5
    fig, ax = plt.subplots(nrows=1, ncols=N_plots, sharex=True)

    for plot_index, it_index in enumerate(np.linspace(1, N_iterations - 1, N_plots)):
        it_index = int(it_index)
        ax[plot_index].hist(positions[:, it_index, 0], bins=int(np.sqrt(N_particles)))
        ax[plot_index].title.set_text(f"it: {it_index}")

    plt.show()


def main_multi():
    # contains the distance from the origin of a given particle on a given iteration
    positions = np.zeros((N_particles, N_iterations, 2))
    with multiprocessing.Pool() as p:
        pos_array = p.imap_unordered(propagate_particle, range(N_particles))
        with tqdm(total=N_particles) as pbar:
            for particle_index, pos in enumerate(
                p.imap_unordered(propagate_particle, range(N_particles))
            ):
                pbar.update()
                positions[particle_index] = pos

    # plot results
    N_plots = 5
    fig, ax = plt.subplots(nrows=1, ncols=N_plots, sharex=True)

    for plot_index, it_index in enumerate(np.linspace(1, N_iterations - 1, N_plots)):
        it_index = int(it_index)
        ax[plot_index].hist(positions[:, it_index, 0], bins=int(np.sqrt(N_particles)))
        ax[plot_index].title.set_text(f"it: {it_index}")

    plt.show()


if __name__ == "__main__":
    main_multi()
