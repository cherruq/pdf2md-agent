"""内部调优策略与用于流水线优化的静态默认常量。"""

from __future__ import annotations

from typing import Final


# 用于跨页拼接(stitch) Markdown 的默认模式 ("heuristic" 或 "off")
DEFAULT_STITCH_MODE: Final[str] = "heuristic"

# 当 AI 提取重试失败时，是否降级/回退(fallback)到原生 PDF 文本层
FALLBACK_TO_TEXT: Final[bool] = True

# Token 预算安全余量 (为安全保留的上下文限制百分比)
TOKEN_BUDGET_SAFETY_DEFAULT: Final[float] = 0.85

# 在迭代降采样期间允许的最小长边像素限制
IMAGE_MIN_LONG_SIDE: Final[int] = 768

# 用于 PDF 页面光栅化的默认渲染 DPI
DEFAULT_DPI: Final[int] = 144

# 默认图像约束
DEFAULT_IMAGE_LONG_SIDE: Final[int] = 1536
DEFAULT_IMAGE_JPEG_QUALITY: Final[int] = 85

# AI 提取覆盖率反思(reflection)参数
REFLECTION_COVERAGE_THRESHOLD: Final[float] = 0.90
REFLECTION_MAX_ATTEMPTS: Final[int] = 2
PENALTY_PROMPT: Final[str] = (
    "\n\nCRITICAL WARNING: Your previous output missed significant portions "
    "of the native text. You MUST preserve ALL text. Please re-read the "
    "page carefully and transcribe completely."
)


__all__ = [
    "DEFAULT_IMAGE_JPEG_QUALITY",
    "DEFAULT_IMAGE_LONG_SIDE",
    "DEFAULT_STITCH_MODE",
    "FALLBACK_TO_TEXT",
    "IMAGE_MIN_LONG_SIDE",
    "PENALTY_PROMPT",
    "REFLECTION_COVERAGE_THRESHOLD",
    "REFLECTION_MAX_ATTEMPTS",
    "TOKEN_BUDGET_SAFETY_DEFAULT",
]
