"""veRL 0.8 的窄范围运行时兼容。"""


def install_torch_padding_fallback():
    """Install the pinned veRL compatibility hooks used by WLX GRPO workers."""
    from verl.utils import attention_utils
    from verl.utils import npu_flash_attn_utils as fallback

    functions = (
        fallback.index_first_axis,
        fallback.pad_input,
        fallback.rearrange,
        fallback.unpad_input,
    )
    # ponytail: veRL 0.8 在 CUDA 上硬导入 FA2；上游提供 torch fallback 后删除此 hook。
    attention_utils._get_attention_functions = lambda: functions

    try:
        from shopping_grpo.training.grpo.wlx_step_grpo import install_wlx_step_grpo
    except ImportError:
        # Lightweight unit tests mock only veRL's padding module. The full
        # training preflight separately requires and verifies this hook.
        return
    install_wlx_step_grpo()
