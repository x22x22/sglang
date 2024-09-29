from __future__ import annotations

"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Meta data for a forward pass."""
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List

import numpy as np
import torch

if TYPE_CHECKING:
    from sglang.srt.layers.attention_backend import AttentionBackend
    from sglang.srt.managers.schedule_batch import ImageInputs, ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner


class ForwardMode(IntEnum):
    # Prefill a new sequence. This is deprecated now. "EXTEND" covers this case.
    PREFILL = auto()
    # Extend a sequence. The KV cache of the first part of the sequence is already computed (e.g., system prompt).
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()
    # Contains both PREFILL and EXTEND.
    MIXED = auto()

    def is_prefill(self):
        return self == ForwardMode.PREFILL

    def is_extend(self):
        return self == ForwardMode.EXTEND or self == ForwardMode.MIXED

    def is_decode(self):
        return self == ForwardMode.DECODE

    def is_mixed(self):
        return self == ForwardMode.MIXED


@dataclass
class InputMetadata:
    """Store all inforamtion of a forward pass."""

    forward_mode: ForwardMode
    batch_size: int
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor

    # Output location of the KV cache
    out_cache_loc: torch.Tensor

    # Position information
    positions: torch.Tensor = None

    # For extend
    extend_seq_lens: torch.Tensor = None
    extend_prefix_lens: torch.Tensor = None
    extend_start_loc: torch.Tensor = None
    extend_no_prefix: bool = None

    # For logprob
    return_logprob: bool = False
    top_logprobs_nums: List[int] = None
    extend_seq_lens_cpu: List[int] = None
    extend_logprob_start_lens_cpu: List[int] = None

    # For multimodal
    image_inputs: List[ImageInputs] = None

    # For attention
    req_to_token_pool: ReqToTokenPool = None
    token_to_kv_pool: BaseTokenToKVPool = None
    attn_backend: AttentionBackend = None

    @classmethod
    def from_schedule_batch(
        cls,
        batch: ScheduleBatch,
    ):
        ret = cls(
            forward_mode=batch.forward_mode,
            batch_size=batch.batch_size(),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            out_cache_loc=batch.out_cache_loc,
            return_logprob=batch.return_logprob,
            top_logprobs_nums=batch.top_logprobs_nums,
        )

        if ret.forward_mode.is_decode():
            ret.positions = (ret.seq_lens - 1).to(torch.int64)
        else:
            ret.positions = torch.tensor(
                np.concatenate(
                    [
                        np.arange(batch.prefix_lens_cpu[i], len(req.fill_ids))
                        for i, req in enumerate(batch.reqs)
                    ],
                    axis=0,
                ),
                device="cuda",
            ).to(torch.int64)

            ret.image_inputs = [r.image_inputs for r in batch.reqs]
            ret.extend_seq_lens = torch.tensor(batch.extend_lens_cpu, device="cuda")
            ret.extend_prefix_lens = torch.tensor(batch.prefix_lens_cpu, device="cuda")
            ret.extend_start_loc = torch.zeros_like(ret.extend_seq_lens)
            ret.extend_start_loc[1:] = torch.cumsum(ret.extend_seq_lens[:-1], dim=0)
            ret.extend_no_prefix = all(x == 0 for x in batch.prefix_lens_cpu)
            ret.extend_seq_lens_cpu = batch.extend_lens_cpu
            ret.extend_logprob_start_lens_cpu = batch.extend_logprob_start_lens_cpu

        return ret
