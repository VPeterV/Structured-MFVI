from supar.utils.fn import kmeans
from supar.utils.data import Sampler

def get_bucket_sampler(lengths, max_tokens, n_buckets, shuffle=True, distributed=False, evaluate=False):
    buckets = dict(zip(*kmeans(lengths, n_buckets)))
    return Sampler(buckets=buckets,
                   batch_size=max_tokens,
                   shuffle=shuffle,
                   distributed=distributed)