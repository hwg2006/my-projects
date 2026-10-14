"""把训练好的 .pt 权重导出为 ONNX，供 Rust / Python(onnxruntime) / 后端服务使用。

用法:
    python training/export_onnx.py --weights runs/train/coco128_yolo11n/weights/best.pt
    python training/export_onnx.py --weights yolo11n.pt --imgsz 640 --out models/yolo11n.onnx
"""

import argparse
import sys
from pathlib import Path

# 以文件方式运行时 sys.path 首项是脚本目录，先补上仓库根目录再导入共享配置
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_config import MODELS_DIR, ROOT  # noqa: E402

from ultralytics import YOLO  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--weights",
        default=str(ROOT / "runs" / "train" / "coco128_yolo11n" / "weights" / "best.pt"),
    )
    p.add_argument("--imgsz", type=int, default=416)
    p.add_argument("--opset", type=int, default=12)
    p.add_argument(
        "--out",
        default=str(MODELS_DIR / "yolo11n_coco128.onnx"),
        help="复制到 models/ 的目标文件名",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    onnx_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,
        dynamic=False,
        half=False,
    )
    print(f"ONNX 已导出: {onnx_path}")

    # 复制一份到 models/ 统一管理
    import shutil

    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, dst)
    print(f"已复制到: {dst}")

    # 关键信息提示后续使用
    anchors = (args.imgsz // 8) ** 2 + (args.imgsz // 16) ** 2 + (args.imgsz // 32) ** 2
    print(f"输入尺寸: {args.imgsz}x{args.imgsz}, 输出 [1, 4+nc, {anchors}]（nc=类别数）")


if __name__ == "__main__":
    main()
