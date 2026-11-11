"""CVDF MNIST test-only cache, strict IDX parsing and preselected image pairs."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import struct
from urllib.request import urlopen
import numpy as np
from scipy.spatial.distance import cdist
from .instance import OTProblem
from .mainstream import weight_stats
from ..io.config import save_json

SOURCE='https://github.com/cvdfoundation/mnist'
BASE='https://storage.googleapis.com/cvdf-datasets/mnist/'
FILES=('t10k-images-idx3-ubyte.gz','t10k-labels-idx1-ubyte.gz')


def read_idx(path):
    with gzip.open(path,'rb') as stream:
        data=stream.read()
    if len(data)<4 or data[:3]!=b'\x00\x00\x08':
        raise ValueError('expected unsigned-byte IDX')
    ndim=data[3]
    if ndim not in (1,3) or len(data)<4+4*ndim:
        raise ValueError('invalid IDX dimensions')
    shape=struct.unpack('>'+'I'*ndim,data[4:4+4*ndim])
    payload=np.frombuffer(data,dtype=np.uint8,offset=4+4*ndim)
    if not all(shape) or payload.size!=int(np.prod(shape)):
        raise ValueError('IDX payload length mismatch')
    return payload.reshape(shape).copy()


def ensure_mnist(cache='data/mnist',download=True):
    root=Path(cache);root.mkdir(parents=True,exist_ok=True)
    info={}
    for filename in FILES:
        path=root/filename
        if not path.exists():
            if not download:
                raise FileNotFoundError('blocked_data: missing '+str(path))
            partial=path.with_suffix('.part')
            with urlopen(BASE+filename,timeout=45) as response,partial.open('wb') as target:
                while True:
                    chunk=response.read(1024*1024)
                    if not chunk:break
                    target.write(chunk)
            read_idx(partial)
            partial.replace(path)
        info[filename]=dict(url=BASE+filename,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            shape=list(read_idx(path).shape))
    images,labels=(read_idx(root/name) for name in FILES)
    if images.shape!=(10000,28,28) or labels.shape!=(10000,) or np.any(labels>9):
        raise ValueError('not the MNIST test split')
    save_json(dict(source=SOURCE,split='test',files=info),root/'provenance.json')
    return images,labels,info


def choose_pairs(labels,seed=20261110):
    generator=np.random.default_rng(seed);used=set();pairs=[]
    specifications=[(f'same_{d}',d,d) for d in (0,2,4,6,8)]+[(f'cross_{d}_{d+1}',d,d+1) for d in (0,2,4,6,8)]
    for name,da,db in specifications:
        indices=[]
        for digit in (da,db):
            eligible=np.array([int(i) for i in np.flatnonzero(labels==digit) if int(i) not in used])
            if not len(eligible):raise ValueError('insufficient unused images')
            index=int(generator.choice(eligible));used.add(index);indices.append(index)
        pairs.append(dict(name=name,indices=indices,labels=[da,db],split='test',
                          pair_type='same' if da==db else 'cross'))
    return dict(seed=seed,selection_rule='listed order, choose one index per side, globally without replacement',pairs=pairs)


def image_problem(images,pair,epsilon,case_id=None,provenance=None):
    if images.ndim!=3 or images.shape[1:]!=(28,28):
        raise ValueError('expected 28x28 images')
    ia,ib=pair['indices'];a0=images[ia].ravel().astype(float);b0=images[ib].ravel().astype(float)
    if a0.sum()<=0 or b0.sum()<=0 or np.any(a0<0) or np.any(b0<0):
        raise ValueError('image must have positive total mass')
    mask_a,mask_b=a0>0,b0>0
    a,b=a0[mask_a]/a0.sum(),b0[mask_b]/b0.sum()
    grid=np.indices((28,28)).reshape(2,-1).T/27.
    x,y=grid[mask_a],grid[mask_b]
    M=cdist(x,y,metric='sqeuclidean')/2.
    cid=case_id or f'M5_{pair["name"]}_e{epsilon:g}'
    metadata=dict(case_id=cid,family='M5',n=len(a),m=len(b),epsilon=epsilon,data_seed=20261110,
        weight_seed_a=None,weight_seed_b=None,marginal_type='image_gray',alpha=None,
        raw_cost_max=2.,normalization_policy='full 28x28 grid diameter squared = 2',
        split='test',indices=[ia,ib],labels=pair['labels'],pair_type=pair['pair_type'],pair_name=pair['name'],
        original_pixels=784,support_n=len(a),support_m=len(b),retained_mass_a=float(a.sum()),
        retained_mass_b=float(b.sum()),raw_gray_mass_a=float(a0.sum()),raw_gray_mass_b=float(b0.sum()),
        data_files=provenance or {},weight_stats_a=weight_stats(a),weight_stats_b=weight_stats(b),marginal_floor=None)
    arrays=dict(x=x,y=y,a=a,b=b,mask_a=mask_a,mask_b=mask_b,image_a=images[ia],image_b=images[ib])
    return OTProblem(a,b,M,epsilon,cid,metadata),arrays
