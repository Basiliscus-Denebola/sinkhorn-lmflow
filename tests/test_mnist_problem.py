import gzip
import struct
import numpy as np
import pytest
from sinkhorn_lmflow.problems.mnist import read_idx,choose_pairs,image_problem,ensure_mnist


def test_idx_fixture_and_invalid_payload(tmp_path):
    p=tmp_path/'sample.gz';data=np.arange(12,dtype=np.uint8).reshape(2,2,3)
    with gzip.open(p,'wb') as f:f.write(struct.pack('>IIII',2051,2,2,3)+data.tobytes())
    np.testing.assert_array_equal(read_idx(p),data)
    with gzip.open(p,'wb') as f:f.write(b'bad')
    with pytest.raises(ValueError):read_idx(p)
    with pytest.raises(FileNotFoundError):ensure_mnist(tmp_path/'missing',download=False)


def test_pairing_and_support_scale():
    labels=np.tile(np.arange(10),10)
    pairs=choose_pairs(labels)
    assert pairs==choose_pairs(labels)
    indices=[i for p in pairs['pairs'] for i in p['indices']]
    assert len(set(indices))==20
    images=np.zeros((2,28,28),dtype=np.uint8)
    images[0,0,0]=1;images[0,27,27]=255;images[1,0,27]=1
    p,arrays=image_problem(images,dict(name='same_0',indices=[0,1],labels=[0,0],pair_type='same'),.01)
    assert p.M.shape==(2,1)
    np.testing.assert_array_equal(p.M,np.full((2,1),.5))
    assert p.a.min()==1/256 and p.metadata['raw_cost_max']==2
    assert arrays['mask_a'].sum()==2 and p.a.sum()==p.b.sum()==1
