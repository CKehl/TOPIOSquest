import numpy as np
import itertools

class ParticleScheduler:
    _cores = 50
    _gdims = []
    _gcdims = []
    _gclayout = []

    def __init__(self):
        self._gdims = [40, 30]
        self._gcdims = [5, 5]
        self._gclayout = [int(self._gdims[0]/self._gcdims[0]), int(self._gdims[1]/self._gcdims[1])]

    def schedule_in_sequence(self, particle_pos_array):
        result = [None] * particle_pos_array.shape[0]
        core_index = 0
        for i in range(0, particle_pos_array.shape[0]):
            result[i] = core_index
            core_index = (core_index+1) % self._cores
        return result

    def schedule_with_grids(self, particle_pos_array):
        result = [None] * particle_pos_array.shape[0]
        total_cells = int(np.prod(np.array(self._gclayout)))
        core_mapper = np.arange(total_cells).reshape(self._gclayout)
        np_gclayout = np.array(self._gclayout)
        for i in range(0, particle_pos_array.shape[0]):
            cell_index = (particle_pos_array[i,:] // np_gclayout).astype(int)
            core_index = core_mapper[cell_index[0],cell_index[1]]
            result[i] = core_index
        return result


if __name__ == "__main__":
    scheduler = ParticleScheduler()
    particles_positions = np.random.rand(100,2)
    particles_positions[:,0] *= 40.0
    particles_positions[:,1] *= 30.0
    core_distribution = scheduler.schedule_in_sequence(particles_positions)
    print(core_distribution)
    core_distribution = scheduler.schedule_with_grids(particles_positions)
    print(core_distribution)