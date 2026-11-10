"""Fixed mainstream EOT inputs, independent geometry/weight RNG streams."""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.spatial.distance import cdist
from .instance import OTProblem

POOL_SIZE = 5000
SOURCE_CENTERS = np.array([[.2,.2],[.8,.2],[.5,.8]])
TARGET_CENTERS = np.array([[.25,.25],[.75,.25],[.5,.75]])


def rng(seed, stream):
    return np.random.default_rng(np.random.SeedSequence([int(seed),int(stream)]))


def array_hash(*arrays):
    digest=hashlib.sha256()
    for value in arrays:
        a=np.ascontiguousarray(value)
        digest.update(str((a.shape,a.dtype.str)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def point_pools(geometry, seed, size=POOL_SIZE):
    if geometry not in ('uniform','mixture'):
        raise ValueError('unknown point geometry')
    pools=[]
    for stream,centers,probs,sigma in ((10,SOURCE_CENTERS,[.5,.3,.2],.06),
                                      (11,TARGET_CENTERS,[.2,.5,.3],.07)):
        gen=rng(seed,stream)
        if geometry=='uniform':
            points=gen.uniform(0,1,(size,2))
        else:
            labels=gen.choice(3,size=size,p=probs)
            points=centers[labels]+gen.normal(0,sigma,(size,2))
        pools.append(points)
    return tuple(pools)


def weights(n, seed, side, alpha=None):
    if alpha is None:
        return np.full(n,1/n)
    if not np.isfinite(alpha) or alpha<=0:
        raise ValueError('positive Dirichlet alpha required')
    value=rng(seed,20 if side=='a' else 21).dirichlet(np.full(n,alpha))
    if not np.isfinite(value).all() or np.any(value<=0):
        raise FloatingPointError('data_generation_failure: zero/nonfinite Dirichlet weight; no floor or resampling')
    return value/value.sum()


def weight_stats(value):
    return dict(min=float(value.min()),max=float(value.max()),
                quantiles=np.quantile(value,[0,.01,.25,.5,.75,.99,1]).tolist())


def make_mainstream(case):
    """Return problem plus compact reconstruction arrays (not one M per run)."""
    family=case['family'];n=int(case['n']);m=int(case.get('m',n));seed=int(case.get('data_seed',42))
    epsilon=float(case['epsilon']);alpha=case.get('alpha')
    if max(n,m)>POOL_SIZE:
        raise ValueError('N=10000 extension is disabled in v1')
    geometry='random' if family=='M1' else 'uniform' if family=='M2' else 'mixture'
    arrays={}
    if geometry=='random':
        raw=rng(seed,1).uniform(0,1,(n,m))
        pool_id=None
    else:
        xpool,ypool=point_pools(geometry,seed)
        x,y=xpool[:n],ypool[:m]
        arrays.update(x=x,y=y)
        pool_id=array_hash(xpool,ypool)
        raw=cdist(x,y,metric='sqeuclidean')
    scale=float(raw.max())
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('degenerate cost scale')
    raw/=scale
    a,b=weights(n,seed,'a',alpha),weights(m,seed,'b',alpha)
    arrays.update(a=a,b=b)
    metadata=dict(case_id=case['case_id'],family=family,n=n,m=m,epsilon=epsilon,data_seed=seed,
        weight_seed_a=[seed,20],weight_seed_b=[seed,21],marginal_type='uniform' if alpha is None else 'dirichlet',
        alpha=alpha,raw_cost_max=scale,normalization_policy='maximum element',geometry=geometry,
        rng_rule='SeedSequence([data_seed, stream]); streams 1=cost,10/11=geometry,20/21=weights',
        point_pool_id=pool_id,point_pool_size=POOL_SIZE if pool_id else None,prefix_n=n,prefix_m=m,
        weight_stats_a=weight_stats(a),weight_stats_b=weight_stats(b),marginal_floor=None)
    return OTProblem(a,b,raw,epsilon,case['case_id'],metadata),arrays


def main_cases():
    cases=[]
    for family in ('M1','M2','M3'):
        for seed in (42,43,44):
            for eps in (.01,.003,.001):
                cases.append(dict(case_id=f'{family}_N1024_s{seed}_e{eps:g}',family=family,n=1024,m=1024,
                                  data_seed=seed,epsilon=eps,alpha=None))
    for seed in (42,43,44):
        for alpha in (1.,.1):
            cases.append(dict(case_id=f'M4_N1024_s{seed}_a{alpha:g}_e0.003',family='M4',n=1024,m=1024,
                              data_seed=seed,epsilon=.003,alpha=alpha))
    for pair in ('same_0','same_2','same_4','same_6','same_8','cross_0_1','cross_2_3','cross_4_5','cross_6_7','cross_8_9'):
        for eps in (.01,.001):
            cases.append(dict(case_id=f'M5_{pair}_e{eps:g}',family='M5',pair_name=pair,n=784,m=784,
                              data_seed=20261110,epsilon=eps,alpha=None))
    for n in (512,2048,5000):
        cases.append(dict(case_id=f'M6_N{n}_s42_e0.003',family='M6',n=n,m=n,data_seed=42,epsilon=.003,alpha=None))
    cases.append(dict(case_id='M7_1024x2048_s42_e0.003',family='M7',n=1024,m=2048,data_seed=42,epsilon=.003,alpha=None))
    assert len(cases)==57
    return cases
