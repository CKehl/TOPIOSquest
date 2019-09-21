import unittest
import numpy as np
import schedule

class MyTestCase(unittest.TestCase):

    def test_ScheduleInSequence(self):
        particles = np.array([[0.9,1.0],[1.8,2.1],[2.7,3.2],[3.6,4.3],[4.5,5.4],
                              [5.4,6.5],[6.3,7.6],[7.2,8.7],[8.1,9.8],[9.0,10.9]])
        scheduler = schedule.ParticleScheduler()
        scheduler.set_cores(5)
        positions = [0,1,2,3,4,0,1,2,3,4]
        self.assertEqual(particles.shape[0], len(positions))
        out_positions = scheduler.schedule_in_sequence(particles)
        self.assertEqual(positions, out_positions)

    def test_ScheduleWithGrid(self):
        particles = np.array([[0.9,1.0],[1.8,2.1],[2.7,3.2],[3.6,4.3],[4.5,5.4],
                              [5.4,6.5],[6.3,7.6],[7.2,8.7],[8.1,9.8],[9.0,10.9]])
        scheduler = schedule.ParticleScheduler()
        scheduler.set_cores(5)
        positions = [0,0,0,0,8,9,9,9,9,17]
        self.assertEqual(particles.shape[0], len(positions))
        out_positions = scheduler.schedule_with_grids(particles)
        self.assertEqual(positions, out_positions)


if __name__ == '__main__':
    unittest.main()
