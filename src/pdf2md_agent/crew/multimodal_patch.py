"""对 CrewAI 的 AddImageTool 进行 Monkey-patch，以 (a) 将本地文件内联为 data URL，
并且 (b) 返回一个 ``VISION_IMAGE`` 哨兵字符串而不是字典。

原生 ``AddImageTool`` 中存在两个 Bug：

1. ``_run(image_url=...)`` 会原封不动地转发 URL。兼容 OpenAI 的视觉
   API 会以 HTTP 400 拒绝本地路径，并将缺失的图像视为
   “未附加图像”，从而导致模型幻觉出页面内容。

2. ``_run`` 返回类似
   ``{"role": "user", "content": [{"type": "image_url", ...}]}`` 的 ``dict``。CrewAI 的
   ``StepExecutor._build_observation_message`` 只知道如何从结果为**字符串**
   ``VISION_IMAGE:<media_type>:<base64>`` 的工具结果中构建
   多模态内容块。字典/字符串化的结果会变成纯文本的 ``Observation: <dict>`` 消息，从而导致图像块丢失。

为了保持在 ``MiniMax-M3`` 约 2k token 的上下文预算内，我们还会重新编码
每个本地图像：通过 ``img.thumbnail(..., LANCZOS)`` 限制长边，
转换为 RGB，写入为 JPEG (``optimize=True``)，然后进行 base64 编码。
上限和质量可以由 runner 在每次调用时配置，默认值为
1536 像素 / 质量 85。

我们替换 ``_run`` 以便在内联编码本地文件后返回正确的哨兵字符串。
``patch_add_image_tool()`` 是幂等的。

线程安全契约 (D15-001/002/003)
----------------------------------------
活动的调整大小/质量控制参数存储在模块全局变量中
(``_patched``, ``_active_long_side``, ``_active_jpeg_quality``)，因此
内部的 ``_run`` 闭包可以在调用时读取它们，而无需显式的
单次调用参数。对 ``patch_add_image_tool()`` 的并发调用和
对修补后的 ``_run`` 的并发调用通过模块级的 ``threading.RLock`` 进行序列化：写入者在临界区内刷新参数，
读取者在委派给 ``_to_sentinel`` 之前在同一锁下对它们进行快照。使用 ``RLock`` (而不是 ``Lock``) 是因为
``patch_add_image_tool`` 可能会在早期初始化路径中被重新进入，
否则会发生自我死锁。

签名故意保持不变 —— 测试会对
``pdf2md_agent.crew.orchestrator.<name>`` 进行打补丁，闭包重构将
使这些补丁目标失效。幂等的安装（只有第一次调用实际安装了修补后的 ``_run``）原样保留。
"""

from __future__ import annotations

import base64
import io
from PIL import UnidentifiedImageError
import logging
import threading
from pathlib import Path

from PIL import Image


log = logging.getLogger("pdf2md_agent.crew.multimodal_patch")

_DEFAULT_TARGET_LONG_SIDE: int = 1536
_DEFAULT_JPEG_QUALITY: int = 85

# 防御性编程：``UnidentifiedImageError`` 是 ``OSError`` 的子类，因此
# 下面的运行时处理程序已经捕获了它。我们显式导入它，以便
# 意图更加清晰；如果导入时 Pillow 缺失，我们回退
# 到 ``OSError``，这样 except 子句仍然可以绑定。
try:
    from PIL import UnidentifiedImageError
except ImportError:  # pragma: no cover
    UnidentifiedImageError = OSError  # type: ignore[assignment,misc]


class ImageEncodeError(RuntimeError):
    """当本地图像无法打开/解码/重新编码时抛出。

    由于内联图像从未进入请求（D6-015），否则 LLM 会对页面内容产生幻觉。
    调用者（CrewAI 步骤执行器/runner）应将此视为可重试
    / 可降级的错误。
    """


def _encode_local_image(
    path: Path,
    *,
    target_long_side: int,
    jpeg_quality: int,
) -> bytes:
    """使用 Pillow 打开 ``path``，缩小尺寸，返回 JPEG 字节。

    如果 Pillow 无法打开或重新编码文件（例如 ``FileNotFoundError``，针对非图像的 ``UnidentifiedImageError`` 等），
    则抛出 :class:`ImageEncodeError`。
    runner 会在 ``layout.pages_dir`` 下预先构建一个缩小尺寸的副本，
    因此此函数几乎不需要做实际工作，但它对于测试中的任意输入仍然是正确的。
    """
    try:
        with Image.open(path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            if target_long_side > 0:
                img.thumbnail((target_long_side, target_long_side), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=jpeg_quality, optimize=True)
            return buf.getvalue()
    except (FileNotFoundError, OSError, UnidentifiedImageError) as exc:
        log.warning(
            "_encode_local_image: cannot encode %s: %s",
            path,
            exc,
        )
        raise ImageEncodeError(f"cannot encode image {path}: {exc}") from exc


def _to_data_url(
    value: str,
    *,
    target_long_side: int = _DEFAULT_TARGET_LONG_SIDE,
    jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
) -> str:
    """将本地文件内联为 ``data:image/jpeg;base64,...`` data URL。

    URL（``http://``, ``https://``, ``data:``）以及无法解析为
    实际文件的路径将原样返回。本地图像在
    内存中通过 Pillow 的 LANCZOS 缩略图算法缩小尺寸，并以
    请求的质量重新编码为 JPEG，然后再进行 base64 编码 —— 生成的 ``data:`` URL
    足够小，可以保留在 ``MiniMax-M3`` 上下文窗口内。

    编码失败会以 :class:`ImageEncodeError` 冒泡抛出，以便
    runner 可以决定是否重试/降级（D6-015）；我们不再
    静默地返回未修改的 ``value``。
    """
    if not value or value.startswith(("http://", "https://", "data:")):
        return value
    path = Path(value)
    if not path.is_file():
        return value
    encoded = _encode_local_image(
        path,
        target_long_side=target_long_side,
        jpeg_quality=jpeg_quality,
    )
    b64 = base64.b64encode(encoded).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _to_sentinel(
    value: str,
    action: str | None,
    *,
    target_long_side: int = _DEFAULT_TARGET_LONG_SIDE,
    jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
) -> str:
    """返回步骤执行器能够识别的 ``VISION_IMAGE:...`` 哨兵。

    哨兵格式为 ``VISION_IMAGE:<media_type>:<base64_data>``。
    我们还在同一字符串中保留了 agent 可选的 action 文本，以便
    执行器的文本降级处理仍能将其呈现给模型。
    """
    url = _to_data_url(
        value,
        target_long_side=target_long_side,
        jpeg_quality=jpeg_quality,
    )
    if not url.startswith("data:"):
        return action or f"(could not inline image at {value})"
    head, b64 = url.split(",", 1)
    media_type = head[len("data:") :].split(";", 1)[0]
    if action:
        return f"{action}\nVISION_IMAGE:{media_type}:{b64}"
    return f"VISION_IMAGE:{media_type}:{b64}"


_patched = False
_active_long_side: int = _DEFAULT_TARGET_LONG_SIDE
_active_jpeg_quality: int = _DEFAULT_JPEG_QUALITY

# 序列化并发写入者（``patch_add_image_tool``）和读取者
# （修补后的 ``_run`` 闭包）。使用 ``RLock`` 以便写入者重新进入
# 自身（例如通过回调此模块的初始化路径）时不会发生死锁。
# 参见上方模块文档字符串中的“线程安全契约”。
_lock = threading.RLock()


def patch_add_image_tool(
    *,
    target_long_side: int = _DEFAULT_TARGET_LONG_SIDE,
    jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
) -> None:
    """包装 ``AddImageTool._run``，使本地路径内联并返回哨兵。

    始终更新活动的调整大小/质量控制参数，以便调用者（CLI、runner）
    可以在模块导入后使用其解析的值重新调用它。
    ``_run`` 补丁本身只安装一次；后续调用仅仅是
    刷新闭包在调用时读取的模块级状态。
    """
    with _lock:
        global _patched, _active_long_side, _active_jpeg_quality
        _active_long_side = target_long_side
        _active_jpeg_quality = jpeg_quality
        if _patched:
            return
        from crewai.tools.agent_tools.add_image_tool import AddImageTool

        def _run(self, image_url: str, action=None, **kwargs):  # type: ignore[override]
            # 在锁下对活动控制参数进行快照，这样我们就永远不会在
            # 更新中途将损坏的 (target, quality) 对传递给 ``_to_sentinel``。
            with _lock:
                target = _active_long_side
                quality = _active_jpeg_quality
            return _to_sentinel(
                image_url,
                action,
                target_long_side=target,
                jpeg_quality=quality,
            )

        AddImageTool._run = _run  # type: ignore[assignment]
        _patched = True
