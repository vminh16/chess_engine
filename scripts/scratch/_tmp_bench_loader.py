import time
import numpy as np
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

REPO = Path(r'C:\Users\USER\Desktop\chess_engine')
DATA_ROOT = REPO / 'data' / 'process' / 'train'
SEED = 123
BATCH_SIZE = 2048
LOCAL_SHUFFLE_BLOCK = 32768
TRAIN_DROP_LAST = True


def _sorted_npy_files(split_dir: Path, pattern: str):
    return sorted(split_dir.glob(pattern), key=lambda p: p.name)


def scan_split(split_dir: Path):
    x_files = _sorted_npy_files(split_dir, 'X_*.npy')
    y_files = _sorted_npy_files(split_dir, 'y_*.npy')
    shard_sizes = []
    for xf, yf in zip(x_files, y_files):
        X = np.load(xf, mmap_mode='r')
        y = np.load(yf, mmap_mode='r')
        shard_sizes.append(int(X.shape[0]))
    shard_sizes = np.array(shard_sizes, dtype=np.int64)
    offsets = np.zeros(len(shard_sizes) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(shard_sizes)
    return {
        'x_files': [str(p) for p in x_files],
        'y_files': [str(p) for p in y_files],
        'shard_sizes': shard_sizes,
        'offsets': offsets,
        'num_shards': len(shard_sizes),
        'num_samples': int(offsets[-1]),
    }


class ShardedNpyDataset(Dataset):
    def __init__(self, meta: dict, dtype_y=torch.float32, use_mmap=True):
        self.x_files = meta['x_files']
        self.y_files = meta['y_files']
        self.offsets = np.asarray(meta['offsets'], dtype=np.int64)
        self.num_samples = int(meta['num_samples'])
        self.num_shards = int(meta['num_shards'])
        self.dtype_y = dtype_y
        self.use_mmap = bool(use_mmap)
        self._X = [None] * self.num_shards
        self._y = [None] * self.num_shards
    def __len__(self):
        return self.num_samples
    def _open_shard_if_needed(self, shard_id: int):
        if self._X[shard_id] is None:
            self._X[shard_id] = np.load(self.x_files[shard_id], mmap_mode='r' if self.use_mmap else None)
        if self._y[shard_id] is None:
            self._y[shard_id] = np.load(self.y_files[shard_id], mmap_mode='r' if self.use_mmap else None)
    def __getitem__(self, idx):
        g = int(idx)
        shard_id = int(np.searchsorted(self.offsets, g, side='right') - 1)
        local_i = int(g - self.offsets[shard_id])
        self._open_shard_if_needed(shard_id)
        return torch.from_numpy(self._X[shard_id][local_i]), torch.as_tensor(self._y[shard_id][local_i], dtype=self.dtype_y)


class ShardLocalBatchSampler:
    def __init__(self, meta, batch_size, drop_last=False, seed=123, shuffle_shards=True, local_shuffle_block=16384, shuffle_within_block=True, shuffle_block_order=True):
        self.offsets = np.asarray(meta['offsets'], dtype=np.int64)
        self.shard_sizes = np.asarray(meta['shard_sizes'], dtype=np.int64)
        self.num_samples = int(meta['num_samples'])
        self.num_shards = int(meta['num_shards'])
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.shuffle_shards = bool(shuffle_shards)
        self.local_shuffle_block = int(local_shuffle_block)
        self.shuffle_within_block = bool(shuffle_within_block)
        self.shuffle_block_order = bool(shuffle_block_order)
        self.epoch = 0
        self.start_batch = 0
    def set_epoch(self, epoch):
        self.epoch = int(epoch)
    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    def _local_order(self, n, rng):
        idx = np.arange(n, dtype=np.int64)
        block_size = max(1, self.local_shuffle_block)
        if block_size <= 1:
            rng.shuffle(idx)
            return idx
        n_blocks = (n + block_size - 1) // block_size
        blocks = np.arange(n_blocks, dtype=np.int64)
        rng.shuffle(blocks)
        out = np.empty(n, dtype=np.int64)
        pos = 0
        for block_id in blocks:
            s = int(block_id * block_size)
            e = min(n, s + block_size)
            block = idx[s:e].copy()
            if self.shuffle_within_block:
                rng.shuffle(block)
            span = e - s
            out[pos:pos+span] = block
            pos += span
        return out
    def _build_global_blocks(self, rng):
        shard_order = np.arange(self.num_shards, dtype=np.int64)
        if self.shuffle_shards:
            rng.shuffle(shard_order)
        block_len = max(self.batch_size, self.local_shuffle_block)
        blocks = []
        for shard_id in shard_order:
            shard_id = int(shard_id)
            n = int(self.shard_sizes[shard_id])
            if n <= 0:
                continue
            start = int(self.offsets[shard_id])
            local_idx = self._local_order(n, rng)
            for local_start in range(0, n, block_len):
                local_end = min(n, local_start + block_len)
                block = local_idx[local_start:local_end]
                if block.size > 0:
                    blocks.append(start + block)
        if self.shuffle_block_order:
            rng.shuffle(blocks)
        return blocks
    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        blocks = self._build_global_blocks(rng)
        carry = []
        for block in blocks:
            block_list = block.tolist()
            i = 0
            n = len(block_list)
            while i < n:
                need = self.batch_size - len(carry)
                j = min(i + need, n)
                carry.extend(block_list[i:j])
                i = j
                if len(carry) == self.batch_size:
                    yield carry
                    carry = []
        if (not self.drop_last) and len(carry) > 0:
            yield carry


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    meta = scan_split(DATA_ROOT)
    ds = ShardedNpyDataset(meta, dtype_y=torch.float32, use_mmap=True)
    sampler = ShardLocalBatchSampler(meta, batch_size=BATCH_SIZE, drop_last=TRAIN_DROP_LAST, seed=SEED + 100, shuffle_shards=True, local_shuffle_block=LOCAL_SHUFFLE_BLOCK, shuffle_within_block=True, shuffle_block_order=True)
    for nw in (0, 2):
        t0 = time.time()
        loader = DataLoader(ds, batch_sampler=sampler, num_workers=nw, pin_memory=False, persistent_workers=(nw > 0), worker_init_fn=seed_worker if nw > 0 else None)
        seen = 0
        nbatches = 0
        ok = True
        err = None
        try:
            for i, (x, y) in enumerate(loader):
                seen += len(y)
                nbatches += 1
                if i >= 29:
                    break
        except Exception as e:
            ok = False
            err = f'{type(e).__name__}: {e}'
        dt = time.time() - t0
        print(f'num_workers={nw} ok={ok} batches={nbatches} seen={seen} time_s={dt:.3f} batches_per_s={(nbatches/dt if dt else 0):.3f} err={err}')
