import numpy as np
import pytest
from scipy.spatial.distance import cdist
from sinkhorn_lmflow.core.cost_profile_distance import generic_distance, geometric_distance, build_distance, _distance_from_gram
from sinkhorn_lmflow.core.semi_dual import center
from sinkhorn_lmflow.problems.instance import OTProblem


def test_gram_direct_variance_translation_and_permutation():
    gen=np.random.default_rng(51)
    M=gen.normal(size=(13,17));a=gen.dirichlet(np.ones(13));eps=.3
    D=generic_distance(M,a,eps)
    expected=np.zeros_like(D)
    for j in range(17):
        for k in range(17):
            diff=M[:,j]-M[:,k]
            expected[j,k]=np.sqrt(a@((diff-a@diff)**2))/eps
    np.testing.assert_allclose(D,expected,atol=1e-13)
    shifted=M+gen.normal(size=(13,1))+gen.normal(size=(1,17))
    np.testing.assert_allclose(generic_distance(shifted,a,eps),D,atol=2e-13)
    perm=gen.permutation(17)
    np.testing.assert_allclose(generic_distance(M[:,perm],a,eps),D[np.ix_(perm,perm)],atol=2e-13)
    assert np.all(np.diag(D)==0)
    Q=np.eye(17)-np.ones((17,17))/17
    assert np.linalg.eigvalsh(Q@D@Q).max()<1e-11
    with pytest.raises(FloatingPointError):
        _distance_from_gram(np.array([[1.,2.],[2.,1.]]))


@pytest.mark.parametrize('mass',[1.,3.])
def test_exact_geometric_nonuniform_rectangular(mass):
    gen=np.random.default_rng(2)
    x=gen.normal(size=(11,2));y=gen.normal(size=(19,2));a=mass*gen.dirichlet(np.ones(11))
    h=3.2;eps=.17;M=cdist(x,y,'sqeuclidean')/h
    np.testing.assert_allclose(geometric_distance(x,y,a,eps,h),generic_distance(M,a,eps),rtol=3e-13,atol=2e-12)
    p=OTProblem(a/mass,np.ones(19)/19,M,eps,metadata=dict(family='M4',raw_cost_max=h))
    D,backend,reason=build_distance(p,dict(x=x,y=y))
    assert backend=='exact_geometric' and reason is None
    np.testing.assert_allclose(D,generic_distance(M,a,eps),atol=2e-12)
    assert build_distance(p,dict(x=x+1,y=y))[1]=='generic_gram'


def test_line_energy_gradient_and_squared_distance_distinction():
    x=np.array([-2.,-.3,.7,2.8]);D=np.abs(x[:,None]-x)
    e=center(np.array([.3,-.2,.5,-.1]));h=center(np.array([.1,.7,-.2,.4]))
    G=lambda v: -.25*(v@D@v)
    assert G(e)==pytest.approx(.5*np.sum(np.diff(x)*np.cumsum(e)[:-1]**2))
    u=center(-.5*D@e)
    assert (G(e+1e-6*h)-G(e-1e-6*h))/2e-6==pytest.approx(u@h,rel=1e-8)
    assert -.25*(e@(D**2)@e)==pytest.approx(.5*(e@x)**2)
    assert abs(G(e)-.5*(e@x)**2)>.01
