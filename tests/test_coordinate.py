"""Fixed examples for boundary optimisation, objective and original KKT rule."""
import unittest
import numpy as np
from scipy.optimize import minimize
from models.baseline import BaselineBasis
from models.features import EventFeatures
from inference.coordinate import refine_likelihood, _coordinate_maximum


class CoordinateTests(unittest.TestCase):
    def test_exact_bound_interior_and_empty_interval(self):
        # log(1+v)-2v is decreasing on v>=0; log(1+v)-v/2 peaks at v=1.
        self.assertEqual(_coordinate_maximum(np.array([1.]), np.array([1.]), 2., 0., 1e-5), 0.)
        self.assertAlmostEqual(_coordinate_maximum(np.array([1.]), np.array([1.]), .5, 0., 0.), 1., places=8)
        f = EventFeatures([], 1, [], 1, .5, 0, 5, include_start=False)
        r = refine_likelihood(f, [[.3]], [[0.]], {}, [[False]])
        self.assertTrue(r['converged'])
        self.assertEqual(r['baseline_coefficients'][0,0], 1e-6)
        self.assertAlmostEqual(r['training_score']['loglik'], -5e-6)

    def test_coordinate_can_reenter_from_zero(self):
        f = EventFeatures([(.1,0),(.11,0),(.12,0)], 1, [], 1, .5, 0, 100, include_start=False)
        r = refine_likelihood(f, [[.03]], [[0.]], {}, [[True]])
        self.assertTrue(r['converged'])
        self.assertGreater(r['alpha_pairwise'][0,0], 0)
        self.assertGreater(r['bound_reentries'], 0)
        self.assertLessEqual(r['kkt_residual'], 1e-4)
        self.assertTrue(np.all(np.diff([r['initial_objective']]+r['objective_history']) >= -1e-6))

    def test_matches_independent_optimizer_with_temporal_background_and_hyperedge(self):
        events = [(.1,0),(.3,1),(.5,0),(.7,1),(1.1,0),(1.15,1),(1.2,0),(2.1,1),(2.2,0),(2.4,1)]
        basis = BaselineBasis('linear', end=3)
        f = EventFeatures(events, 2, [(0,1)], 1, .5, 0, 3, basis, False)
        b = np.full((2,2), .5); a = np.array([[0.,.1],[.1,0.]]); h = {(0,1):.1}
        r = refine_likelihood(f, b, a, h, ~np.eye(2,dtype=bool), kkt_tol=1e-6)
        def unpack(x):
            return x[:4].reshape(2,2), np.array([[0.,x[4]],[x[5],0.]]), {(0,1):x[6]}
        def objective(x):
            bb,aa,hh = unpack(x)
            stats = f.statistics(bb,aa,hh)
            grad = np.r_[stats['background_score'].ravel(),stats['pairwise_score'][0,1],
                         stats['pairwise_score'][1,0],stats['hyperedge_score']]
            return -f.score(bb,aa,hh)['loglik'], -grad
        independent = minimize(objective, [.5]*4+[.1]*3, jac=True, method='L-BFGS-B',
                               bounds=[(1e-6,None)]*4+[(0.,None)]*3,
                               options={'ftol':1e-14,'gtol':1e-9,'maxiter':1000})
        self.assertTrue(independent.success)
        self.assertTrue(r['converged'])
        self.assertAlmostEqual(r['final_objective'], -independent.fun, places=5)
        self.assertTrue(np.all(np.diag(r['alpha_pairwise']) == 0))
        second = EventFeatures(events+[(4.,1)], 2, [(0,1)], 1, .5, 0, 3, basis, False)
        same = refine_likelihood(second, b, a, h, ~np.eye(2,dtype=bool), kkt_tol=1e-6)
        np.testing.assert_array_equal(r['baseline_coefficients'], same['baseline_coefficients'])
        np.testing.assert_array_equal(r['alpha_pairwise'], same['alpha_pairwise'])


if __name__ == '__main__':
    unittest.main()
