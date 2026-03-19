import torch
import torch.nn as nn

class GhostBatchNorm2d(nn.Module):
    DEFAULT_SPLITS = 1

    def __init__(self, num_features, splits=None, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.splits = splits if splits is not None else self.DEFAULT_SPLITS
        
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(
                num_features, 
                eps=eps, 
                momentum=momentum, 
                affine=affine, 
                track_running_stats=track_running_stats
            ) 
            for _ in range(self.splits)
        ])

    def forward(self, x):
        if self.training:
            B = x.size(0)
            if B % self.splits != 0:
                raise ValueError(
                    f"GhostBatchNorm2d: Expected batch size to be cleanly divisible "
                    f"by splits ({self.splits}), but got batch size {B}."
                )
            
            chunks = x.chunk(self.splits, dim=0)
            normalized_chunks = [self.bns[i](chunk) for i, chunk in enumerate(chunks)]
            return torch.cat(normalized_chunks, dim=0)
        else:
            return self.bns[0](x)
