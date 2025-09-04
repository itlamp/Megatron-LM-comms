# Copyright (C) 2025 Intel Corporation
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import MoEAlltoAllSEQTokenDispatcher
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_lazy_mode, is_real_cuda_device_available


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig, layer_number: int = None):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"

        assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(MoELayer, self).__init__(config=config, layer_number=layer_number)
        self.moe_layer_recompute = config.moe_layer_recompute

        self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.ep_size = parallel_state.get_expert_model_parallel_world_size()

        # Initialize router
        self.router = TopKRouter(config=self.config)

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall_seq":
            self.token_dispatcher = MoEAlltoAllSEQTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == None:
            self.token_dispatcher = None
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        self.experts = build_module(self.submodules.experts, self.num_local_experts, self.config)

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(self.submodules.shared_experts, config=self.config)
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

    def _checkpointed_forward(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor
    ):
        def custom_forward(hidden_states, probs, routing_map):
            (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
                hidden_states, probs, routing_map
            )
            expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
            output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
            if self.use_shared_expert and not self.shared_expert_overlap:
                # if shared_expert_overlap is True, the expert calculation happens in
                # the token_dispatcher to overlap communications and computations
                output += self.shared_experts(hidden_states)
            return output, mlp_bias

        def custom_forward_hpu(hidden_states, probs, routing_map):
            tp_ep_group = parallel_state.get_expert_tensor_and_model_parallel_group()
            if self.tp_size > 1 or self.ep_size > 1:
                with torch.no_grad():
                    global_routing_map = tensor_parallel.gather_from_sequence_parallel_region(
                        routing_map, group=tp_ep_group
                    )
                global_probs = tensor_parallel.gather_from_sequence_parallel_region(
                    probs, group=tp_ep_group
                )
                global_hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                    hidden_states, group=tp_ep_group, use_global_buffer=is_lazy_mode()
                )
            else:
                global_hidden_states = hidden_states
                global_probs = probs
                global_routing_map = routing_map

            # process tokens with expert MLPs
            expert_output, mlp_bias = self.experts(
                global_hidden_states, global_probs, global_routing_map
            )

            if self.tp_size > 1 or self.ep_size > 1:
                output = tensor_parallel.reduce_scatter_to_sequence_parallel_region(
                    expert_output, group=tp_ep_group
                )
            else:
                output = expert_output

            output = output.view(hidden_states.shape)

            if self.use_shared_expert and not self.shared_expert_overlap:
                # if shared_expert_overlap is True, the expert calculation happens in
                # the token_dispatcher to overlap communications and computations
                output += self.shared_experts(hidden_states)

            return output, mlp_bias

        fn = custom_forward_hpu if self.config.moe_dynamic_hpu else custom_forward
        if self.moe_layer_recompute:
            if self.config.fp8 and not is_real_cuda_device_available():
                from intel_transformer_engine.distributed import checkpoint as te_checkpoint

                output, mlp_bias = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                    probs,
                    routing_map,
                )
            else:
                if is_real_cuda_device_available() or is_lazy_mode():
                    output, mlp_bias = tensor_parallel.checkpoint(
                        fn, False, hidden_states, probs, routing_map
                    )
                else:
                    output, mlp_bias = torch.utils.checkpoint.checkpoint(
                        fn, hidden_states, probs, routing_map, use_reentrant=True
                    )
        else:
            output, mlp_bias = fn(hidden_states, probs, routing_map)

        return output, mlp_bias

    def forward(self, hidden_states: torch.Tensor):
        if (
            self.training
            and self.config.tensor_model_parallel_size > 1
            and not self.config.sequence_parallel
        ):
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        probs, routing_map = self.router(hidden_states)
        output, mlp_bias = self._checkpointed_forward(hidden_states, probs, routing_map)

        return output, mlp_bias
