import numpy as np

from .data import IGNORE, load_mask


class Oracle:
    def __init__(self, ids, masks, root):
        self._masks = dict(zip(ids, masks))
        self.root = root
        self.labeled = set()
        self.log = []

    def query(self, ids):
        ids = list(ids)
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate ids in one query")
        repeated = [i for i in ids if i in self.labeled]
        if repeated:
            raise ValueError(f"already labeled: {repeated[:5]}")
        unknown = [i for i in ids if i not in self._masks]
        if unknown:
            raise KeyError(f"not in pool: {unknown[:5]}")
        self.labeled.update(ids)
        self.log.append(ids)
        return np.stack([self._masks[i] for i in ids])

    def cost_proxy(self, ids):
        """Post-hoc annotation cost (analysis only): object instances and object-boundary pixels."""
        n_objects, boundary = 0, 0
        for i in ids:
            obj = load_mask(self.root, i, kind="SegmentationObject")
            n_objects += len(set(np.unique(obj).tolist()) - {0, IGNORE})
            cls = load_mask(self.root, i).astype(np.int32)
            pad = np.pad(cls, 1, mode="edge")
            differs = np.zeros_like(cls, dtype=bool)
            for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
                differs |= pad[dy:dy + cls.shape[0], dx:dx + cls.shape[1]] != cls
            boundary += int((differs & (cls != 0) & (cls != IGNORE)).sum())
        return n_objects, boundary
