"""Standing feature bits and checked raw contexts, without Python oracle replay.

The independent oracle remains a mandatory launch gate. Training still hashes
every shard and matches its row identity to hash-checked document windows. This
adapter reuses exactly the two native decoders already validated for evaluation;
it discards only unused raw Python row objects and redundant feature rebuilding.
"""
from .c14_b_eval_decode import eval_shard,load_context_decoder
_context_decoder=None

def decode(data,count,decoder=None):
    global _context_decoder
    if _context_decoder is None:_context_decoder=load_context_decoder()
    identities,lengths,values,contexts=eval_shard(data,count,decoder,_context_decoder)
    return identities,lengths,values,contexts,None
