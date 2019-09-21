import numpy as np
import itertools
import time

def print_timing (func):
  def wrapper (*arg):
    t1 = time.time ()
    res = func (*arg)
    t2 = time.time ()
    print ("{0} took {1:.4f} ms".format (func.__name__, (t2 - t1) * 1000.0))
    return res

  return wrapper

class ParticleScheduler:
    """
    Particles are given as (x,y) sequence. Grid is given in (x,y)-sequence (see self._gdims, self._gcdims, self._gclayout).
    Thus, the grid-based scheduler needs to translate (row, colum) into (x,y).
    """
    _cores = 50
    _gdims = []
    _gcdims = []
    _gclayout = []

    def __init__(self):
        self._gdims = [40, 30]
        self._gcdims = [5, 5]
        self._gclayout = [int(self._gdims[0]/self._gcdims[0]), int(self._gdims[1]/self._gcdims[1])]

    def set_cores(self, cores):
        self._cores = cores

    def get_cores(self):
        return self._cores

    def schedule_in_sequence(self, particle_pos_array, cores_k):
        self.set_cores(cores_k)
        self.schedule_in_sequence(particle_pos_array)

    def schedule_with_grids(self, particle_pos_array, cores_k):
        self.set_cores(cores_k)
        self.schedule_with_grids(particle_pos_array)

    def schedule_in_sequence(self, particle_pos_array):
        if np.min(particle_pos_array[:,0]<40.0).astype(bool) is False:
            return []
        if np.min(particle_pos_array[:,0]>0).astype(bool) is False:
            return []
        if np.min(particle_pos_array[:,1]<30.0).astype(bool) is False:
            return []
        if np.min(particle_pos_array[:,1]>0).astype(bool) is False:
            return []
        return self.schedule_in_sequence_baseline(particle_pos_array)

    def schedule_with_grids(self, particle_pos_array):
        if np.min(particle_pos_array[:,0]<40.0).astype(bool) is False:
            return []
        if np.min(particle_pos_array[:,0]>0).astype(bool) is False:
            return []
        if np.min(particle_pos_array[:,1]<30.0).astype(bool) is False:
            return []
        if np.min(particle_pos_array[:,1]>0).astype(bool) is False:
            return []
        return self.schedule_with_grids_baseline(particle_pos_array)

    @print_timing
    def schedule_in_sequence_baseline(self, particle_pos_array):
        result = [None] * particle_pos_array.shape[0]
        core_index = 0
        for i in range(0, particle_pos_array.shape[0]):
            result[i] = core_index
            core_index = (core_index+1) % self._cores
        return result

    @print_timing
    def schedule_with_grids_baseline(self, particle_pos_array):
        result = [None] * particle_pos_array.shape[0]
        total_cells = int(np.prod(np.array(self._gclayout)))
        np_gclayout = np.array(self._gclayout)
        core_mapper = np.arange(total_cells).reshape(np.flip(np_gclayout,0))
        cell_indices = (particle_pos_array // self._gcdims).astype(int)[:,[1,0]]
        for i in range(0, particle_pos_array.shape[0]):
            core_index = core_mapper[cell_indices[i,0],cell_indices[i,1]]
            result[i] = core_index
        return result

    @print_timing
    def schedule_in_sequence_islice(self, particle_pos_array):
        result = [None] * particle_pos_array.shape[0]
        core_index = 0
        for i in itertools.islice(itertools.count(), 0, particle_pos_array.shape[0]):
            result[i] = core_index
            core_index = (core_index+1) % self._cores
        return result

    @print_timing
    def schedule_with_grids_islice(self, particle_pos_array):
        result = [None] * particle_pos_array.shape[0]
        total_cells = int(np.prod(np.array(self._gclayout)))
        np_gclayout = np.array(self._gclayout)
        core_mapper = np.arange(total_cells).reshape(np.flip(np_gclayout,0))
        cell_indices = (particle_pos_array // self._gcdims).astype(int)[:,[1,0]]
        for i in itertools.islice(itertools.count(), 0, particle_pos_array.shape[0]):
            core_index = core_mapper[cell_indices[i,0],cell_indices[i,1]]
            result[i] = core_index
        return result

    @print_timing
    def schedule_in_sequence_npforelem(self, particle_pos_array):
        result = [None] * particle_pos_array.shape[0]
        core_index = 0
        for i, particle in enumerate(particle_pos_array):
            result[i] = core_index
            core_index = (core_index+1) % self._cores
        return result

    @print_timing
    def schedule_with_grids_npforelem(self, particle_pos_array):
        result = [None] * particle_pos_array.shape[0]
        total_cells = int(np.prod(np.array(self._gclayout)))
        np_gclayout = np.array(self._gclayout)
        core_mapper = np.arange(total_cells).reshape(np.flip(np_gclayout,0))
        cell_indices = (particle_pos_array // self._gcdims).astype(int)[:,[1,0]]
        for i, cell_index in enumerate(cell_indices):
            core_index = core_mapper[cell_index[0],cell_index[1]]
            result[i] = core_index
        return result

    @print_timing
    def schedule_in_sequence_npndindex(self, particle_pos_array):
        result = [None] * particle_pos_array.shape[0]
        core_index = 0
        for i in np.ndindex(particle_pos_array.shape[0]):
            result[i[0]] = core_index
            core_index = (core_index+1) % self._cores
        return result

    @print_timing
    def schedule_with_grids_npndindex(self, particle_pos_array):
        result = [None] * particle_pos_array.shape[0]
        total_cells = int(np.prod(np.array(self._gclayout)))
        np_gclayout = np.array(self._gclayout)
        core_mapper = np.arange(total_cells).reshape(np.flip(np_gclayout,0))
        cell_indices = (particle_pos_array // self._gcdims).astype(int)[:,[1,0]]
        for i in np.ndindex(cell_indices.shape[0]):
            core_index = core_mapper[cell_indices[i[0],0],cell_indices[i[0],1]]
            result[i[0]] = core_index
        return result


if __name__ == "__main__":
    scheduler = ParticleScheduler()
    simple_particles_positions = np.random.rand(1000000,2)
    simple_particles_positions[:,0] *= 40.0
    simple_particles_positions[:,1] *= 30.0
    print("In-Sequence scheduling - Baseline")
    scheduler.schedule_in_sequence_baseline(simple_particles_positions)
    print("In-Sequence scheduling - Itertools.ISlice")
    scheduler.schedule_in_sequence_islice(simple_particles_positions)
    print("In-Sequence scheduling - Numpy for-loop over array")
    scheduler.schedule_in_sequence_npforelem(simple_particles_positions)
    print("In-Sequence scheduling - Numpy for-loop via np.ndindex")
    scheduler.schedule_in_sequence_npndindex(simple_particles_positions)
    print("In-Cell scheduling - Baseline")
    scheduler.schedule_with_grids_baseline(simple_particles_positions)
    print("In-Cell scheduling - Itertools.ISlice")
    scheduler.schedule_with_grids_islice(simple_particles_positions)
    print("In-Cell scheduling - Numpy for-loop over array")
    scheduler.schedule_with_grids_npforelem(simple_particles_positions)
    print("In-Cell scheduling - Numpy for-loop over via np.ndindex")
    scheduler.schedule_with_grids_npndindex(simple_particles_positions)


    print("Final run with 10^8 particles ...")
    particles_positions = np.random.rand(100000000,2)
    core_distribution = scheduler.schedule_in_sequence(particles_positions)
    core_distribution = scheduler.schedule_with_grids(particles_positions)