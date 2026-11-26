"""为评测数据提供跨平台、稳定的文本内容指纹。"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_text_file(path: Path) -> str:
    """将 LF 和 CRLF 统一为旧冻结记录使用的 CRLF 后再计算 SHA-256。"""
    text = path.read_text(encoding="utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\r\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
