import unittest
import numpy as np
from scipy.optimize import minimize
from models.baseline import BaselineBasis
from models.features import EventFeatures
from models.kernel import ExponentialKernel,HyperedgeAnchor
from models.diagnostics import stability_diagnostics
from inference.e_step import EStep
from inference.m_step import MStep
from inference.em import run_em
from models.tensor_param import HypergraphTensor


class FreeEMTests(unittest.TestCase):
    def test_one_iteration_matches_original(self):
        events = [(.1,0),(.3,1),(.6,0),(.8,1),(1.1,0),(1.4,1)]
        kernel,anchor = ExponentialKernel(1),HyperedgeAnchor(.5)
        mu = np.array([.3,.4]); a = np.array([[.1,.2],[.15,.1]]); h = {(0,1):.2}
        r = EStep(kernel,anchor).compute(events,mu,a,h,[(0,1)],target_start=.5,target_end=2)
        m = MStep(2,HypergraphTensor(2,rank=2,seed=0),lambda_l1=.1)
        expected_mu = m.update_mu(events,r['p_background'],2,target_start=.5,target_end=2)
        expected_a = m.update_alpha_pairwise(events,r['p_pairwise'],r['p_hyper'],[(0,1)],kernel,2,target_start=.5,target_end=2)
        expected_h = m.update_alpha_hyper(events,r['p_hyper'],[(0,1)],anchor,kernel,2,target_start=.5,target_end=2)
        result = run_em(events,2,2,[(0,1)],kernel,anchor,mu,a,h,n_iter=1,lambda_l1=.1,
                        hyper_update='free',target_start=.5,target_end=2,track_loglik=True)
        np.testing.assert_allclose(result.mu,expected_mu,atol=1e-12)
        np.testing.assert_allclose(result.alpha_pairwise,expected_a,atol=1e-12)
        self.assertAlmostEqual(result.alpha_hyper[(0,1)],expected_h[(0,1)],places=12)

    def test_trend_solution_matches_direct_optimization_and_no_leakage(self):
        events = [(.1,0),(.4,0),(.8,0),(1.8,0),(2.2,0),(2.8,0)]
        basis = BaselineBasis('linear',end=4)
        kw = dict(T=4,n_nodes=1,edge_list=[],kernel=ExponentialKernel(1),anchor_calc=HyperedgeAnchor(.5),
                  mu0=np.array([1.]),alpha_pairwise0=np.zeros((1,1)),alpha_hyper0={},lambda_l1=0,
                  hyper_update='free',baseline_basis=basis,pairwise_mask=np.zeros((1,1),dtype=bool),
                  target_start=0,target_end=3,include_start=False,positive_start=True,
                  max_iter=1000,max_total_iter=5000,kkt_tol=1e-4,convergence_tol=1e-9,track_loglik=True,assert_ascent=True)
        result = run_em(events=events,**kw)
        self.assertTrue(result.converged)
        self.assertLessEqual(result.kkt_residual,1e-4)
        f = EventFeatures(events,1,[],1,.5,0,3,basis,False)
        opt = minimize(lambda x:-f.score(x[None,:],np.zeros((1,1)),{})['loglik'],[1.,1.],
                       method='L-BFGS-B',bounds=[(1e-6,None)]*2,
                       options={'ftol':1e-13,'gtol':1e-9})
        self.assertTrue(opt.success)
        self.assertAlmostEqual(result.final_loglik,-opt.fun,places=5)
        second = run_em(events=events+[(3.2,0),(3.8,0)],**kw)
        np.testing.assert_array_equal(result.mu,second.mu)
        np.testing.assert_array_equal(result.alpha_pairwise,np.zeros((1,1)))
        self.assertTrue(np.all(np.diff(result.objective_history)>=-1e-6))

    def test_positive_allowed_starts_and_stability_labels(self):
        events=[(.1,0),(.2,1),(.3,0),(.4,1),(.5,0),(.6,1)]
        r = run_em(events,1,2,[(0,1)],ExponentialKernel(1),HyperedgeAnchor(.5),np.ones(2),
                   np.zeros((2,2)),{(0,1):0},n_iter=1,hyper_update='free',positive_start=True,
                   pairwise_mask=~np.eye(2,dtype=bool))
        self.assertGreater(r.alpha_pairwise[0,1],0)
        self.assertGreater(r.alpha_hyper[(0,1)],0)
        np.testing.assert_array_equal(np.diag(r.alpha_pairwise),0)
        d = stability_diagnostics([.3,.4],np.eye(2)*1.1,{(0,1):.2},1)
        self.assertFalse(d['stationary_existence_sufficient'])
        self.assertNotIn('explosive',d)


if __name__ == '__main__':unittest.main()
