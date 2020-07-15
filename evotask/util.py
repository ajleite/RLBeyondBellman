''' Basic utility functions. '''

import typing

def _extract(seq, i):
    if not isinstance(seq, typing.Sequence):
        return seq[i]

    return [_extract(e, i) for e in seq]

class extract(typing.Sequence):
    def __init__(self, seq):
        self.seq = seq
    def __len__(self):
        return len(self.seq)
    def __getitem__(self, key):
        return _extract(self.seq, key)
