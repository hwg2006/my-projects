# 项目合集（Monorepo）

本仓库汇总两个独立项目，按顶级目录严格区分、互不依赖：

| 目录 | 项目 | 简介 | 独立仓库 |
| --- | --- | --- | --- |
| [`yolo-realtime/`](./yolo-realtime) | YOLO 实时目标检测 | YOLO11n 训练 → ONNX 导出 → Python/Rust 推理 → FastAPI 服务 → Vue3 前端 / Electron 桌面端，纯 CPU 实时检测 | [hwg2006/yolo-realtime](https://github.com/hwg2006/yolo-realtime) |
| [`xin-qing-navigation/`](./xin-qing-navigation) | 华农心晴导航 | AI 心理导航应用（server / web / desktop / cli / docker） | [hwg2006/hnau-xinqing-navigation](https://github.com/hwg2006/hnau-xinqing-navigation) |

## 说明

- 两个目录各自保留完整的 git 提交历史（经 `git subtree` 合并，历史中路径自动带上目录前缀）。
- 各项目的技术栈、启动方式、工程日志见各自目录内的 README / 工程日志。
- 两个项目的持续开发仍在上述独立仓库进行，本仓库为汇总视图；如需在子目录里继续开发，可使用 `git subtree pull/push` 与独立仓库双向同步。
