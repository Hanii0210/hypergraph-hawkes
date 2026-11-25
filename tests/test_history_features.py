import unittest
import numpy as np
from scipy.integrate import quad
from models.baseline import BaselineBasis, TemporalBaseline
from models.features import EventFeatures
from models.kernel import ExponentialKernel, HyperedgeAnchor
from models.likelihood import log_likelihood_interval
from inference.e_step import EStep
from inference.m_step import MStep
from models.tensor_param import HypergraphTensor


EVENTS = [(-.4,0),(-.2,1),(0.,0),(.1,1),(.1,0),(.6,0),(.75,1),
          (1.2,0),(1.2,1),(1.9,2),(2.1,0),(2.4,1),(2.8,2)]
EDGES = [(0,1),(0,1,2),(0,2)]
MU = np.array([.3,.4,.5])
A = np.array([[.03,.12,.08],[.07,.02,.09],[.11,.04,.01]])
H = {(0,1):.25,(0,1,2):.15,(0,2):.09}


def direct(t,node,events,baseline,a,h,beta=1.3,delta=.5):
    """Deliberately slow definition, independent of the recurrence and caches."""
    history = [(s,v) for s,v in events if s<t]
    lam = baseline.values([t])[0,node]
    lam += sum(a[v,node]*np.exp(-beta*(t-s)) for s,v in history)
    for edge,weight in h.items():
        if node not in edge:continue
        completions = [c for c,v in history if v in edge and all(
            any(w==member and c-delta<=s<=c for s,w in history) for member in edge)]
        if completions:lam += weight*np.exp(-beta*(t-max(completions)))
    return lam


class HistoryFeatureTests(unittest.TestCase):
    def setUp(self):
        self.kernel,self.anchor = ExponentialKernel(1.3),HyperedgeAnchor(.5)

    def test_ties_history_and_independent_compensator(self):
        basis = BaselineBasis("linear",end=3)
        bg = TemporalBaseline([[.2,.5],[.3,.6],[.4,.7]],basis)
        f = EventFeatures(EVENTS,3,EDGES,1.3,.5,0,3,basis,False)
        ref = [direct(t,n,EVENTS,bg,A,H) for t,n in zip(f.times,f.nodes)]
        np.testing.assert_allclose(f.intensity(bg.coefficients,A,H),ref,atol=1e-12)
        cuts = sorted(set([0.,3.]+[t for t,n in EVENTS if 0<t<3]))
        integral = sum(quad(lambda t:sum(direct(t,n,EVENTS,bg,A,H) for n in range(3)),l,r,epsabs=1e-10)[0]
                       for l,r in zip(cuts[:-1],cuts[1:]))
        self.assertAlmostEqual(f.score(bg.coefficients,A,H)["integral"],integral,places=8)
        whole = log_likelihood_interval(EVENTS,0,3,bg,A,H,EDGES,self.kernel,self.anchor,include_start=False)
        split = sum(log_likelihood_interval(EVENTS,l,r,bg,A,H,EDGES,self.kernel,self.anchor,include_start=False)
                    for l,r in [(0,1.2),(1.2,3)])
        self.assertAlmostEqual(whole,split,places=11)

    def test_statistics_match_dense_responsibilities(self):
        f = EventFeatures(EVENTS,3,EDGES,1.3,.5,.05,3)
        stat = f.statistics(MU,A,H)
        dense = EStep(self.kernel,self.anchor).compute(EVENTS,MU,A,H,EDGES,target_start=.05,target_end=3)
        m = MStep(3,HypergraphTensor(3,rank=2,seed=1),lambda_l1=.03)
        mb = m.update_mu(EVENTS,dense["p_background"],3,target_start=.05,target_end=3)
        ma = m.update_alpha_pairwise(EVENTS,dense["p_pairwise"],dense["p_hyper"],EDGES,self.kernel,3,target_start=.05,target_end=3)
        mh = m.update_alpha_hyper(EVENTS,dense["p_hyper"],EDGES,self.anchor,self.kernel,3,target_start=.05,target_end=3)
        np.testing.assert_allclose(stat["background_counts"][:,0]/2.95,mb,atol=1e-12)
        np.testing.assert_allclose(stat["pairwise_counts"]/f.pair_exposure[:,None],ma,atol=1e-12)
        np.testing.assert_allclose(stat["hyperedge_counts"]/(f.hyper_exposure+.03),[mh[e] for e in EDGES],atol=1e-12)

    def test_latest_anchor_replaces_and_future_is_ignored(self):
        events = [(0.1,0),(.2,1),(.3,0),(.4,1)]
        f = EventFeatures(events,2,[(0,1)],1,.5,0,1)
        self.assertAlmostEqual(f.hyper[-1,0],np.exp(-.1))
        f2 = EventFeatures(events+[(2.,0)],2,[(0,1)],1,.5,0,1)
        np.testing.assert_array_equal(f.hyper,f2.hyper)
        np.testing.assert_array_equal(f.pair,f2.pair)
        # Adding an event cannot revoke any already completed pattern.
        increased = sorted(events+[(.25,1)])
        q = EventFeatures(increased,2,[(0,1)],1,.5,0,1)
        for i,t in enumerate(f.times):
            j = np.flatnonzero(q.times==t)[0]
            self.assertGreaterEqual(q.hyper[j,0],f.hyper[i,0]-1e-14)

    def test_linear_storage_and_zero_test_events(self):
        events = [(i/100.,i%3) for i in range(1000)]
        f = EventFeatures(events,3,EDGES,1,.5,0,10,include_start=False)
        self.assertLess(f.nbytes,1000*200)
        empty = EventFeatures(EVENTS,3,EDGES,1.3,.5,3,4)
        parts = empty.score(MU,A,H)
        self.assertEqual(parts["n_events"],0)
        self.assertEqual(parts["event_log_intensity"],0)
        self.assertLess(parts["loglik"],0)


if __name__ == "__main__":unittest.main()
