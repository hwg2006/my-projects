"""项目级配置：仓库路径、运行环境与权重解析，供所有 Python 入口共用。

ultralytics 需要在被导入前就拿到 `YOLO_CONFIG_DIR`，因此本模块在被导入时即完成设置。
`server` 包直接 `from project_config import ...` 即可（服务从仓库根目录启动）；
`training/` 与 `inference_python/` 下的脚本以文件方式运行，其 `sys.path` 首项是脚本
所在目录，需先手动把仓库根目录加入 `sys.path` 再导入本模块。
"""

from __future__ import annotations

import os
from pathlib import Path

# 仓库根目录（本文件所在目录）
ROOT = Path(__file__).resolve().parent

# 让 ultralytics 的配置/字体缓存落在工作区内，避免写入用户目录被沙箱拦截
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))

MODELS_DIR = ROOT / "models"
# 官方 yolo11n 预训练权重（全 COCO 11.8 万张图 / 80 类）：轻量、CPU 单帧 ≈55ms，作为默认
OFFICIAL_PT = ROOT / "yolo11n.pt"
OFFICIAL_ONNX = MODELS_DIR / "yolo11n.onnx"
# 自训练（COCO128 微调）产出，识别能力较弱，仅显式指定时使用
DEFAULT_PT = ROOT / "runs" / "train" / "coco128_yolo11n" / "weights" / "best.pt"
DEFAULT_ONNX = MODELS_DIR / "yolo11n_coco128.onnx"


def weights_by_priority() -> str:
    """按优先级选择权重：官方 n（ONNX 优先于 .pt）> 自训练 > ultralytics 默认。

    注：官方 yolo11s 识别更强但 CPU 单帧约 175ms（n 版约 55ms），实时框滞后明显，
    故不作为默认；需要时可通过各入口的 `--weights models/yolo11s.onnx` 显式指定。
    """
    for p in (OFFICIAL_ONNX, OFFICIAL_PT, DEFAULT_ONNX, DEFAULT_PT):
        if p.exists():
            return str(p)
    return "yolo11n.pt"
