"""
export_rkllm.py — Qwen2.5 转 RKLLM 格式

用法:
    python export_rkllm.py <input_model_dir> <output.rkllm>

依赖:
    pip install rkllm-toolkit

RKLLM 转换流程:
    1. HuggingFace / ModelScope 导出的 Qwen2.5-0.5B-Instruct HF 格式
    2. INT8 量化 → 极致压缩 (从 ~1GB → ~500MB)
    3. 输出 .rkllm 文件, 由 librkllm.so 在板端加载推理
"""

import sys
import os


def export_to_rkllm(input_dir: str, output_path: str):
    """
    将 HF 格式的模型转换为 RKLLM 格式

    Args:
        input_dir:  HuggingFace 模型目录 (含 config.json, tokenizer, safetensors 等)
        output_path: 输出的 .rkllm 文件路径
    """
    try:
        from rkllm import RKLLM  # type: ignore
    except ImportError:
        print("[ERROR] 请先安装 rkllm-toolkit: pip install rkllm-toolkit")
        sys.exit(1)

    if not os.path.isdir(input_dir):
        print(f"[ERROR] 模型目录不存在: {input_dir}")
        sys.exit(1)

    print(f"[RKLLM] 加载模型: {input_dir}")
    llm = RKLLM()

    # 配置参数
    llm.config.num_npu_core = 3          # RK3588 三核 NPU 全开
    llm.config.max_context_len = 512     # 对话上下文长度
    llm.config.max_new_tokens = 128      # 最大生成 token 数
    llm.config.quant = "w8a8"            # INT8 量化

    print("[RKLLM] 开始转换 + 量化 (INT8)...")
    ret = llm.export(
        model_path=input_dir,
        output_path=output_path,
        config=llm.config,
    )

    if ret == 0:
        print(f"[RKLLM] 导出成功 → {output_path}")
    else:
        print(f"[RKLLM] 导出失败! ret={ret}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export_rkllm.py <hf_model_dir> <output.rkllm>")
        print("Example: python export_rkllm.py ./Qwen2.5-0.5B-Instruct ./qwen2.5_0.5b.rkllm")
        sys.exit(1)

    export_to_rkllm(sys.argv[1], sys.argv[2])
