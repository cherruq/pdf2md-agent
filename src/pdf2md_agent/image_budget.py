"""每页图片预算规划器。

Vision API 会拒绝总 token 数量超过配置的上下文窗口的请求。
:func:`plan_for_image` 针对页面 PNG 挑选出一个最小的缩放比例 ``long_side``，
使得调用仍然保持在 ``ctx_limit * safety`` 限制以下 —— 即所能容纳的最大尺寸，
从而尽可能保留 OCR 还原度。

实现分为两层：

* :class:`BudgetDecision` — 强类型结果，不可变的冻结数据类。
* :func:`plan_for_image` — 入口点。内部使用 :func:`_open_for_size`（当
  Pillow 无法打开文件时使用基于字节大小的回退方案）以及在 ``long_side``
  上执行二分查找。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from PIL import Image  # type: ignore[import-not-found]
from pathlib import Path
from typing import Final

log = logging.getLogger("pdf2md_agent.image_budget")

# Shared with :mod:`pdf2md_agent.token_estimator`. Imported lazily so the
# image-budget planner can be tested without the estimator pulling in
# extra dependencies.
_BYTES_PER_TOKEN: Final[float] = 3.5
"""每个 token 对应的 base64 字符数 —— 从 ``token_estimator`` 镜像过来；
规划器用它进行基于大小的估算，而不依赖评估器的公开 API。"""


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    """单词提取调用的预算结果。

    属性:
        total: 使用所选 ``needed_long_side`` 时 persona + fixed_text + 图像的 token 数总和。
            调整后的大小在安全余量内。
        limit: ``ctx_limit * TOKEN_BUDGET_SAFETY`` (向下取整)。
        fits: 如果所选方案保持调用在 ``limit`` 内，则为 ``True``。
            ``False`` 意味着即使允许最小程度的缩小加上固定文本的预算，仍然超过
            ``limit`` —— 调用者应记录警告并以最小尺寸继续执行。
        needed_long_side: 页面图像在内联之前应该缩放到的长边长度（以像素为单位）。
            始终设置：即使图像已经能放下，该值依然默认为请求的
            ``target_long_side`` 以保持运行具有可预测性。
        reason: 解释决定的简短的、人类可读的文字（用于日志）。
    """

    total: int
    limit: int
    fits: bool
    needed_long_side: int
    reason: str


# --- Size estimation primitives --------------------------------------------


def _b64_chars(size_bytes: int) -> int:
    """返回对 ``size_bytes`` 原始字节进行编码的 base64 字符串的长度。"""
    return ((size_bytes + 2) // 3) * 4


def _tokens_for_size(size_bytes: int) -> int:
    """将原始字节数转换为其估计的 base64 token 消耗。"""
    return max(1, math.ceil(_b64_chars(size_bytes) / _BYTES_PER_TOKEN))


def _est_size_at_long_side(
    orig_bytes: int,
    orig_long_side: int,
    target_long_side: int,
) -> int:
    """粗略的像素面积模型：字节数以 ``(target/orig_long_side)²`` 的比例进行缩放。

    JPEG 文件并不严格遵循这一点，但偏差在两个方向上都是保守的 —— 文字密集型
    页面大致压缩到相当于 PNG 的大小，而照片页面压缩得略多一些。这对挑选
    long_side 来说已经足够好；毕竟 LLM 只看到最终生成的字节数。
    """
    if orig_long_side <= 0 or orig_bytes <= 0:
        return orig_bytes
    scale_sq = (target_long_side / orig_long_side) ** 2
    return max(1024, int(orig_bytes * scale_sq))


def _bytes_to_fallback_size(num_bytes: int) -> tuple[int, int]:
    """从 JPEG 的字节数估算 ``(long_side, long_side)``。

    启发式规则：quality-85 的 JPEG 压缩后大约 0.25 字节/像素
    （≈ 4 像素/字节），所以对于正方形图像而言，``long_side ≈ sqrt(bytes * 4)``
    ≈ ``2 * sqrt(bytes)``。以 256 像素为底限，以确保对于 0 字节或者极小的文件
    仍可生成正常的尺寸，而不是产生会被二分查找以 ``orig_long_side <= min_long_side`` 拒绝的 0 值。
    """
    if num_bytes <= 0:
        return 256, 256
    side = max(256, int(2 * math.sqrt(num_bytes)))
    return side, side


def _open_for_size(path: Path, *, fallback_bytes: int) -> tuple[int, int]:
    """使用 Pillow 返回 ``path`` 的 ``(width, height)``。

    当 Pillow 无法打开文件时，退回到基于字节计算估算。
    如果没有这种回退机制，规划器会将损坏的图像当作 1×1 大小处理，
    从而得出它足以放进预算的错误结论，并将这个超大的原始数据块
    内联至 LLM（缺陷 D6-007）。``fallback_bytes`` 是由调用者缓存的
    ``Path.stat().st_size``，因此这一估算能够如实反映损坏的数据块
    到底包含了多少数据量。
    """

    try:
        with Image.open(path) as img:
            return img.size
    except Exception as exc:
        log.warning(
            "plan_for_image: cannot open %s to read size (%s); using "
            "bytes-based fallback (%d bytes) to keep planner sizing sane",
            path,
            exc,
            fallback_bytes,
        )
        return _bytes_to_fallback_size(fallback_bytes)


# --- Public entry point -----------------------------------------------------


def plan_for_image(
    ctx_limit: int,
    *,
    persona_tokens: int,
    fixed_text_tokens: int,
    image_path: Path,
    target_long_side: int = 1536,
    min_long_side: int = 768,
    jpeg_quality: int = 85,
    safety: float = 0.85,
) -> BudgetDecision:
    """规划在预算内提取所调用的单页面图像调整大小。

    此决定在 ``[min_long_side, original_long_side]`` 范围内寻找**最大的**
    ``long_side``，以保持 *估计的* 图像 token 数以及其他文本始终低于
    ``ctx_limit * safety`` （即，适应的最温和的缩小比例）。如果原图已经能够适应，
    ``needed_long_side`` 会依旧被报告为配置的 ``target_long_side`` 以保持
    运行时的行为一致性。

    参数:
        ctx_limit: 模型所报告的硬性上下文窗口 token 上限
            （关于如何解析，请参见 :func:`pdf2md_agent.config.resolve_ctx_limit`）。
        persona_tokens: 对于智能体人设 + 任务系统提示词的估计 token 数
            （调用者应预先计算完毕）。
        fixed_text_tokens: 对每页变量的估计 token 数：包含
            可选的 text-hint 以及渲染过的任务描述骨架。
        image_path: 页面 PNG 的本地路径。
        target_long_side: 当预算允许时期望缩小到的长边长度；
            这是通常的“happy path”尺寸。
        min_long_side: 二分查找的下界 —— 绝不推荐缩放到比此值
            还小的尺寸（OCR 清晰度的下限）。
        jpeg_quality: 当前评估器未使用此值；已为未来每像素字节的
            校准预留位置。
        safety: 我们愿意使用的 ``ctx_limit`` 的比例；它必须位于 ``(0.0, 1.0]`` 区间。

    返回:
        一个具有反映所选的 long_side 能否使得调用控制在安全限额之内的 ``fits`` 标志的
        :class:`BudgetDecision`。
    """
    del jpeg_quality  # 预留；参阅文档字符串
    if not (0.0 < safety <= 1.0):
        raise ValueError(f"safety must be in (0, 1], got {safety!r}")
    if target_long_side < 1 or min_long_side < 1:
        raise ValueError(
            f"target_long_side and min_long_side must be >= 1, got target={target_long_side}, min={min_long_side}"
        )
    if target_long_side < min_long_side:
        raise ValueError(f"target_long_side={target_long_side} < min_long_side={min_long_side}")

    limit = int(ctx_limit * safety)
    budget_for_image = max(0, limit - persona_tokens - fixed_text_tokens)
    try:
        size = image_path.stat().st_size
    except OSError:
        size = 0
    original_bytes = size
    current_tokens = _tokens_for_size(size) if size > 0 else 0

    if current_tokens <= budget_for_image:
        total = persona_tokens + fixed_text_tokens + current_tokens
        return BudgetDecision(
            total=total,
            limit=limit,
            fits=True,
            needed_long_side=target_long_side,
            reason="image fits within budget at original size",
        )

    orig_w, orig_h = _open_for_size(image_path, fallback_bytes=original_bytes)
    orig_long_side = max(orig_w, orig_h)

    if orig_long_side <= min_long_side:
        total = persona_tokens + fixed_text_tokens + current_tokens
        return BudgetDecision(
            total=total,
            limit=limit,
            fits=False,
            needed_long_side=orig_long_side,
            reason=(f"image already smaller than min_long_side={min_long_side}; fixed-text budget is too tight"),
        )

    upper = orig_long_side
    best = min_long_side
    low, high = min_long_side, upper
    while high - low > 4:
        mid = (low + high) // 2
        est_bytes = _est_size_at_long_side(original_bytes, upper, mid)
        est_tokens = _tokens_for_size(est_bytes)
        if est_tokens <= budget_for_image:
            best = mid
            low = mid
        else:
            high = mid

    if best < upper:
        est_bytes = _est_size_at_long_side(original_bytes, upper, best)
        est_tokens = _tokens_for_size(est_bytes)
        total = persona_tokens + fixed_text_tokens + est_tokens
        return BudgetDecision(
            total=total,
            limit=limit,
            fits=total <= limit,
            needed_long_side=best,
            reason=(f"downscaled from {upper}px to {best}px to fit budget={budget_for_image} image tokens"),
        )

    est_bytes = _est_size_at_long_side(original_bytes, upper, target_long_side)
    est_tokens = _tokens_for_size(est_bytes)
    total = persona_tokens + fixed_text_tokens + est_tokens
    return BudgetDecision(
        total=total,
        limit=limit,
        fits=total <= limit,
        needed_long_side=target_long_side,
        reason="binary search exhausted budget headroom; using target_long_side",
    )


__all__ = ["BudgetDecision", "plan_for_image"]
