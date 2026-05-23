from torch.utils.data import Sampler
import torch.distributed as dist
import torch


class BucketSampler(Sampler):
    def __init__(
        self,
        bucket_indexs,
        aspect_ratios,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        self.bucket_indexs = bucket_indexs
        self.aspect_ratios = aspect_ratios
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.num_replicas = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        self.rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        # 每个bucket内部shuffle后切batch
        all_batches = []
        for ratio, indices in self.bucket_indexs.items():
            if self.shuffle:
                indices = [indices[i] for i in torch.randperm(len(indices), generator=g).tolist()]
            
            all_batches.extend([
                indices[i : i + self.batch_size * self.num_replicas][self.rank :: self.num_replicas]
                for i in range(0, len(indices), self.batch_size * self.num_replicas)
                if len(indices[i : i + self.batch_size * self.num_replicas]) == self.batch_size * self.num_replicas
            ])

        # 随机打乱batch顺序
        for i in torch.randperm(len(all_batches), generator=g).tolist():
            yield all_batches[i]

    def __len__(self):
        total = sum(
            len(v) // (self.batch_size * self.num_replicas)
            for v in self.bucket_index.values()
        )
        return total
