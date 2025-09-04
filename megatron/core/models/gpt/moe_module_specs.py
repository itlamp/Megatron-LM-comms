# Copyright (C) 2025 Intel Corporation
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import warnings
from typing import Optional

from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import (
    GroupedMLP,
    IntelDynamicMLP,
    SequentialMLP,
    TEGroupedMLP,
)
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.utils import get_te_version, is_te_min_version
from megatron.core.version_utils import is_habana_frameworks_min_version

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TEColumnParallelLinear,
        TERowParallelGroupedLinear,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


try:
    from megatron.core.extensions.intel_transformer_engine import (
        IntelTEColumnParallelGroupedLinear,
        IntelTEColumnParallelLinear,
        IntelTERowParallelGroupedLinear,
        IntelTERowParallelGroupedLinearFP8Disabled,
        IntelTERowParallelLinear,
        IntelTERowParallelLinearFp8Disabled,
    )
except:
    pass


def get_moe_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    fp8_coverage: dict = {},
    moe_dynamic_hpu: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
    assert num_experts is not None

    linear_fc1 = None
    linear_fc2 = None
    if use_te:
        if HAVE_TE:
            linear_fc1 = TEColumnParallelLinear
            linear_fc2 = TERowParallelLinear
        else:
            linear_fc1 = IntelTEColumnParallelLinear
            linear_fc2 = (
                IntelTERowParallelLinear
                if fp8_coverage.get('mlp_row_parallel', True)
                else IntelTERowParallelLinearFp8Disabled
            )

    mlp = MLPSubmodules(
        linear_fc1=linear_fc1 if use_te else ColumnParallelLinear,
        linear_fc2=linear_fc2 if use_te else RowParallelLinear,
    )

    # experts spec
    if moe_grouped_gemm:
        ## use GroupedMLP
        if (
            use_te
            and (
                (HAVE_TE and TEColumnParallelGroupedLinear is not None)
                or (not HAVE_TE and IntelTEColumnParallelGroupedLinear is not None)
            )
            and not moe_use_legacy_grouped_gemm
        ):
            ## use TEGroupedLinear
            expert_module = TEGroupedMLP
            iterpgl = None
            if not HAVE_TE:
                iterpgl = (
                    IntelTERowParallelGroupedLinear
                    if fp8_coverage.get('mlp_row_parallel', True)
                    else IntelTERowParallelGroupedLinearFP8Disabled
                )
            expert_submodule = MLPSubmodules(
                linear_fc1=(
                    TEColumnParallelGroupedLinear if HAVE_TE else IntelTEColumnParallelGroupedLinear
                ),
                linear_fc2=TERowParallelGroupedLinear if HAVE_TE else iterpgl,
            )
        else:
            ## use legacy GroupedMLP
            expert_module = GroupedMLP
            expert_submodule = None
            warnings.warn(
                'The legacy GroupedMLP will be deprecated in Megatron-Core v0.12.0. '
                'Please update the TransformerEngine to version>=1.7.0 and use TEGroupedMLP.'
            )
    elif moe_dynamic_hpu and not HAVE_TE:
        expert_module = IntelDynamicMLP
        expert_submodule = None
    else:
        ## use SequentialMLP
        expert_module = SequentialMLP
        if use_te and not (
            is_te_min_version("1.7.0.dev0") or is_habana_frameworks_min_version("1.21.0")
        ):
            warnings.warn(
                "Only transformer-engine>=1.7.0 supports MoE experts, "
                f"but your version is {get_te_version()}. Use local linear implementation instead."
            )
            expert_submodule = MLPSubmodules(
                linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear
            )
        else:
            expert_submodule = mlp

    experts = ModuleSpec(module=expert_module, submodules=expert_submodule)

    # shared experts spec
    shared_experts = ModuleSpec(module=SharedExpertMLP, params={"gate": False}, submodules=mlp)

    # MoE module spec
    moe_module_spec = ModuleSpec(
        module=MoELayer, submodules=MoESubmodules(experts=experts, shared_experts=shared_experts)
    )
    return moe_module_spec
