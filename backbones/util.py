import torch
import numpy as np
import random


def masks_like(tensor, zero=False, generator=None, p=0.2):
    # assert isinstance(tensor, list)
    out1 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    out2 = [torch.ones(u.shape, dtype=u.dtype, device=u.device) for u in tensor]

    if zero:
        if generator is not None:
            for u, v in zip(out1, out2):
                random_num = torch.rand(
                    1, generator=generator, device=generator.device).item()
                if random_num < p:
                    u[0, :] = torch.normal(
                        mean=-3.5,
                        std=0.5,
                        size=(1,),
                        device=u.device,
                        generator=generator).expand_as(u[0, :]).exp()
                    v[0, :] = torch.zeros_like(v[0, :])
                else:
                    u[0, :] = u[0, :]
                    v[0, :] = v[0, :]
        else:
            for u, v in zip(out1, out2):
                u[0, :] = torch.zeros_like(u[0, :])
                v[0, :] = torch.zeros_like(v[0, :])

    return out1, out2
