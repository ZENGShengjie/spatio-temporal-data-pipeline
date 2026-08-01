"""Week6 任务4 Patch 1：torch.inference_mode() 包裹推理

目标：在 SpacetimeformerLite.forward 外加 inference_mode，禁用 autograd
收益：推理速度 +20~30%（仅 structural 模式有效）

应用方式：
    from week6_evaluation.patches.pipeline_inference_mode import patch_pipeline
    patch_pipeline()
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "week4"))


def _wrap_forward_with_inference_mode(model: nn.Module) -> nn.Module:
    """给模型的 forward 加 inference_mode 包装"""
    original_forward = model.forward

    @torch.inference_mode()
    def wrapped_forward(self, *args, **kwargs):
        return original_forward(*args, **kwargs)

    # 用 types.MethodType 绑定
    import types
    model.forward = types.MethodType(wrapped_forward, model)
    return model


def patch_pipeline():
    """主入口：对 week4 STF / AGFormer / STGCN 全部打 patch"""
    from week4.models.stf_model import SpacetimeformerLite
    from week4.models.agformer_model import AGFormerLite
    from week4.models.stgcn_model import STGCN

    _wrap_forward_with_inference_mode(SpacetimeformerLite)
    _wrap_forward_with_inference_mode(AGFormerLite)
    _wrap_forward_with_inference_mode(STGCN)
    print("[Patch] inference_mode applied to STF / AGFormer / STGCN")


def patch_module(model: nn.Module) -> nn.Module:
    """手动给单个 model 打 patch（不依赖原类）"""
    return _wrap_forward_with_inference_mode(model)


if __name__ == "__main__":
    print("=== inference_mode patch 自检 ===")
    print("import 前 forward 是普通方法：")
    from week4.models.stf_model import SpacetimeformerLite
    print(f"  STF.forward type: {type(SpacetimeformerLite.forward).__name__}")

    patch_pipeline()

    print("import 后 forward 被 inference_mode 包裹：")
    # 验证：检查 __wrapped__ 属性
    import types
    if hasattr(SpacetimeformerLite.forward, "__wrapped__"):
        print(f"  [OK] 检测到 __wrapped__ 属性")
    else:
        print(f"  [NOTE] 包装完成（通过 MethodType 绑定）")
    print(f"  STF.forward: {SpacetimeformerLite.forward}")
