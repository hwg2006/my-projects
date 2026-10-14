"""服务配置：默认推理参数与 CORS 白名单。

路径与权重解析统一由仓库根目录的 `project_config.py` 提供。
"""

from __future__ import annotations

import os

# 推理输入尺寸：必须与导出 ONNX 时使用的固定尺寸一致（models/yolo11*.onnx 均为 640）
IMGSZ = 640
# 置信度阈值：越低召回越高（更能找出目标），但误检也越多。
# 目标是「别漏人」（有人进入画面必须框到），coco128 含>=2人的 39 张图实测：
#   conf 0.15→0.10：漏人图数 12→5，检出总数 250→307（重复框 4→10，可接受）。
CONF = float(os.environ.get("YOLO_CONF", "0.10"))
# NMS 的 IoU 阈值：重叠度高于该值的同类框会被合并（非极大值抑制）。
# 注意：收紧该值会把「同一画面里部分重叠的两个人」当重复框抑制掉——这正是「已经框住一个大的人、
# 第二个人进来却识别不到」的原因。coco128 多人图实测（conf=0.15）漏人图数：
#   iou 0.50→16/39、0.60→13、0.70→12、0.80→9；故回到 ultralytics 默认 0.70，优先保召回。
IOU = float(os.environ.get("YOLO_IOU", "0.70"))


def allowed_origins() -> list[str]:
    """CORS 白名单：仅放行本地开发与桌面端来源。

    - Vite dev server（127.0.0.1 / localhost 的 5173）
    - Electron 以 file:// 加载页面时，浏览器发送的 Origin 为字面量 "null"

    可用环境变量 YOLO_ALLOWED_ORIGINS（逗号分隔）整体覆盖。
    """
    env = os.environ.get("YOLO_ALLOWED_ORIGINS")
    if env:
        return [o.strip() for o in env.split(",") if o.strip()]
    return ["http://127.0.0.1:5173", "http://localhost:5173", "null"]


ALLOWED_ORIGINS = allowed_origins()
