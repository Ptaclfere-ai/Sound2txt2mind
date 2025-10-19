# Voice → Summary → Mindmap

> 一键上手：实时转写 → DeepSeek 总结 → 思维导图（Markmap）

## 目录
```
voice-mindmap-app/
  backend/
    main.py
    requirements.txt
    .env.example
  frontend/
    index.html
```

## 后端启动
```bash
cd backend
python -m venv .venv && .venv\Scripts\activate   # Windows PowerShell
pip install -r requirements.txt
copy .env.example .env   # 然后编辑 .env 填入 DEEPSEEK_API_KEY
uvicorn main:app --reload --port 8000
```

## 前端打开
- 用 Chrome 双击 `frontend/index.html`

## 使用
- 左侧【识别模式】
  - 快速演示（浏览器内置）：麦克风实时字幕；
  - 离线精度（Whisper）：可抓取“标签页音频 + 麦克风”，5s 切片上传。
- 右侧点击【生成总结 + 提纲】得到会议纪要与思维导图。

## 提示
- 抓取标签页音频：选择 Whisper 模式，开始后在弹窗中选择具体标签页并勾选“共享音频”。
- 降低延迟：将前端 MediaRecorder 的 `rec.start(5000)` 改小（如 2500/1000）。
- 提升精度：`.env` 里把 `WHISPER_MODEL` 提升为 `medium` 或 `large-v3`（需要更强机器）。
