import unittest
import numpy as np
from scipy.integrate import quad
from models.baseline import BaselineBasis, TemporalBaseline


class TemporalBaselineTests(unittest.TestCase):
    def test_analytic_integrals_and_global_bounds(self):
        bases = [BaselineBasis(),BaselineBasis("sinusoidal",period=5),
                 BaselineBasis("linear",origin=1,end=4),BaselineBasis("quadratic",origin=1,end=4)]
        for basis in bases:
            for a,b in [(-2,0),(0,3),(2,6),(-2,7)]:
                points = [x for x in [1,4] if a<x<b]
                ref = [quad(lambda t:basis.values([t])[0,k],a,b,points=points,epsabs=1e-11)[0] for k in range(basis.size)]
                np.testing.assert_allclose(basis.integral(a,b),ref,atol=1e-9,rtol=1e-9)
            rates = np.array([.3,.7])
            bg = TemporalBaseline(basis.constant_coefficients(rates),basis)
            np.testing.assert_allclose(bg.values(np.linspace(-2,7,101)),np.tile(rates,(101,1)),atol=1e-12)
            coeff = np.arange(1,1+2*basis.size).reshape(2,basis.size)/10
            bg = TemporalBaseline(coeff,basis)
            self.assertTrue(np.all(bg.values(np.linspace(-10,10,1001)) <= bg.upper_bound()+1e-12))
            floor = TemporalBaseline(basis.lower_bounds(2),basis)
            np.testing.assert_allclose(floor.values([-1,0,2,10]),1e-6,atol=1e-16)

    def test_invalid_inputs(self):
        for fn in [lambda:BaselineBasis("sinusoidal",period=0),lambda:BaselineBasis("linear",end=-1),
                   lambda:TemporalBaseline([-1]),lambda:TemporalBaseline([[np.nan]])]:
            with self.assertRaises(ValueError):fn()


if __name__ == "__main__":unittest.main()
