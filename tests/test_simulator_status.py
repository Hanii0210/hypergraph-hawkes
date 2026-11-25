import unittest
import numpy as np
from models.baseline import BaselineBasis,TemporalBaseline
from models.kernel import ExponentialKernel,HyperedgeAnchor
from simulation.simulator import HawkesSimulator,SimulationTruncatedError


class SimulatorStatusTests(unittest.TestCase):
    def test_time_varying_bound_and_reference_intensity(self):
        for basis,c in [(BaselineBasis('sinusoidal',period=3),[[.8,.2],[.4,.6]]),
                        (BaselineBasis('linear',end=8),[[.2,1.2],[.4,.8]])]:
            sim = HawkesSimulator(TemporalBaseline(c,basis),np.array([[.05,.1],[.08,.02]]),
                                  {(0,1):.3},ExponentialKernel(1),HyperedgeAnchor(.5))
            r = sim.simulate(8,seed=12,max_events=1000,burn_in=2,return_info=True,record_trace=True)
            self.assertTrue(r.completed)
            self.assertEqual(r.actual_end,8)
            self.assertGreater(r.n_proposals,0)
            self.assertTrue(all(-2<t<=8 for t,n in r.events))
            for row in r.proposal_trace:
                reference = float(sim._intensity(row['time'],r.events).sum())
                self.assertAlmostEqual(row['intensity'],reference,places=10)
                self.assertLessEqual(row['ratio'],1+1e-12)
            same = sim.simulate(8,seed=12,max_events=1000,burn_in=2,return_info=True)
            self.assertEqual(r.events,same.events)

    def test_cap_is_not_silently_accepted(self):
        sim = HawkesSimulator([100],np.zeros((1,1)),{},ExponentialKernel(1),HyperedgeAnchor(.5))
        r = sim.simulate(10,seed=1,max_events=1,return_info=True)
        self.assertFalse(r.completed)
        self.assertEqual(r.status,'event_cap')
        self.assertLess(r.actual_end,10)
        with self.assertRaises(SimulationTruncatedError):sim.simulate(10,seed=1,max_events=1)

    def test_zero_intensity_completes_empty_path(self):
        sim = HawkesSimulator([0],np.zeros((1,1)),{},ExponentialKernel(1),HyperedgeAnchor(.5))
        r = sim.simulate(2,return_info=True)
        self.assertTrue(r.completed)
        self.assertEqual(r.events,[])


if __name__ == '__main__':unittest.main()
