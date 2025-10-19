import os
import json
import re
import tempfile
import subprocess
import shutil
from typing import Optional

import requests
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from starlette.staticfiles import StaticFiles
from starlette.responses import FileResponse

# ---------- 环境 ----------
load_dotenv()
app = FastAPI(title="Voice→Summary→Mindmap API", version="1.6.0")

# 同源访问即可，不强依赖 CORS；保留以兼容本地调试
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*",],   # 上线同源即可，如需更严谨可改成你的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")

# ---------- 静态资源与首页 ----------
# 将 frontend 作为静态目录挂载到 /static
app.mount("/static", StaticFiles(directory="frontend", html=True), name="static")

# 根路径直接返回前端首页
@app.get("/")
def root_index():
    return FileResponse("frontend/index.html")

# ---------- 小工具 ----------
def _strip_code_fences(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"```[a-zA-Z0-9_-]*\s*([\s\S]*?)\s*```", text)
    return (m.group(1) if m else text).strip()

def _extract_json_block(s: str):
    if not s:
        return None
    try:
        j = json.loads(s)
        if isinstance(j, dict) and ("summary" in j or "outline_md" in j):
            return j
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        try:
            j = json.loads(m.group(0))
            if isinstance(j, dict) and ("summary" in j or "outline_md" in j):
                return j
        except Exception:
            pass
    return None

def _autogen_outline_from_text(txt: str, lang: str = "zh") -> str:
    txt = (txt or "").strip()
    if not txt:
        return "# 会议纪要大纲\n- （暂无内容）"
    if lang == "zh":
        parts = re.split(r"[。！？\n]+", txt)
    else:
        parts = re.split(r"[\.!\?\n]+", txt)
    items = [p.strip(" 　\t-•*") for p in parts if p.strip()]
    items = items[:10] if items else ["（无要点）"]
    out = ["# 会议纪要大纲", "## 要点"]
    out += [f"- {it}" for it in items]
    return "\n".join(out)

def _is_repetitive_zh(text: str) -> bool:
    if not text:
        return False
    s = re.sub(r"\s+", "", text)
    if len(s) < 8:
        return False
    for n in range(2, 7):
        grams = [s[i:i+n] for i in range(0, len(s)-n+1)]
        if not grams:
            continue
        cnt = 0
        i = 0
        while i < len(grams)-1:
            if grams[i] == grams[i+1]:
                cnt += 1
                i += 1
                if cnt >= 3:
                    return True
            else:
                cnt = 0
                i += 1
    return False

# ---------- 健康 ----------
@app.get("/health")
def health():
    import importlib.util
    whisper_installed = importlib.util.find_spec("faster_whisper") is not None
    ffmpeg_found = (os.path.isfile(FFMPEG_BIN)) or (shutil.which(FFMPEG_BIN) is not None)
    return {
        "ok": True,
        "whisper_installed": whisper_installed,
        "ffmpeg": ffmpeg_found,
        "ffmpeg_bin": FFMPEG_BIN,
        "has_api_key": bool(DEEPSEEK_API_KEY),
        "model": WHISPER_MODEL,
        "device": WHISPER_DEVICE,
        "compute_type": WHISPER_COMPUTE_TYPE,
    }

# ---------- Whisper ----------
_asr_model = None
def get_asr_model():
    global _asr_model
    if _asr_model is None:
        try:
            from faster_whisper import WhisperModel
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"未安装 faster-whisper：{e}")
        _asr_model = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE
        )
    return _asr_model

def ffmpeg_to_wav(src_path: str, wav_path: str, ffmpeg_bin: str):
    base = [ffmpeg_bin, "-nostdin", "-y", "-hide_banner", "-loglevel", "error"]
    common_out = ["-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000", wav_path]
    attempts = [
        base + ["-fflags", "+genpts", "-i", src_path] + common_out,
        base + ["-fflags", "+genpts", "-f", "webm", "-i", src_path] + common_out,
        base + ["-fflags", "+genpts", "-f", "matroska", "-i", src_path] + common_out,
        base + ["-fflags", "+genpts", "-f", "ogg", "-i", src_path] + common_out,
    ]
    errs = []
    for cmd in attempts:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and os.path.exists(wav_path) and os.path.getsize(wav_path) > 44:
            return True, None
        errs.append(f"rc={proc.returncode}; {(proc.stderr or '').strip()[:400]}")
        try:
            if os.path.exists(wav_path): os.remove(wav_path)
        except Exception:
            pass
    return False, " | ".join(errs[:4])

def wav_duration_sec(path: str) -> float:
    try:
        import wave, contextlib
        with contextlib.closing(wave.open(path, "rb")) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 1
            return frames / float(rate)
    except Exception:
        return -1.0

@app.post("/asr/chunk")
async def asr_chunk(
    file: UploadFile = File(...),
    lang: Optional[str] = Form(None),
    mime: Optional[str] = Form(None)
):
    try:
        orig_name = file.filename or "chunk.bin"
        suffix = os.path.splitext(orig_name)[-1] or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            src_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"无法读取上传音频：{e}")

    ext = (os.path.splitext(src_path)[-1] or "").lower()
    wav_path = None
    try:
        if ext == ".wav":
            wav_path = src_path
        else:
            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            ok, ferr = ffmpeg_to_wav(src_path, wav_path, FFMPEG_BIN)
            if not ok:
                return {"text": "", "detail": f"ffmpeg 转码失败: {ferr}", "hint": f"frontend_mime={mime or 'unknown'}"}
    except Exception as e:
        try:
            os.remove(src_path)
        finally:
            if wav_path and os.path.exists(wav_path) and wav_path != src_path:
                os.remove(wav_path)
        return {"text": "", "detail": f"转码异常: {e}", "hint": f"frontend_mime={mime or 'unknown'}"}

    try:
        model = get_asr_model()
        language = (lang or "").strip() or None
        dur = wav_duration_sec(wav_path)

        def decode(beam_size=5, temperature=0.0, best_of=None):
            kwargs = dict(
                language=language,
                task="transcribe",
                vad_filter=False,
                beam_size=beam_size,
                temperature=temperature,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.35,
            )
            if best_of is not None:
                kwargs["best_of"] = best_of
            segs, info = model.transcribe(wav_path, **kwargs)
            txt = "".join(s.text for s in segs) or ""
            return txt

        text = decode(beam_size=5, temperature=0.0)
        if not text or _is_repetitive_zh(text):
            text = decode(beam_size=1, temperature=0.6, best_of=5)
        if not text or _is_repetitive_zh(text):
            text = decode(beam_size=1, temperature=0.8, best_of=5)

        if not text:
            return {"text": "", "detail": "ASR empty after decode", "wav_duration": dur}

        if _is_repetitive_zh(text):
            return {"text": text, "detail": "ASR repetitive pattern detected", "wav_duration": dur}

        return {"text": text}

    except Exception as e:
        return {"text": "", "detail": f"ASR error after wav: {e}"}
    finally:
        try:
            if src_path and os.path.exists(src_path):
                if wav_path != src_path:
                    os.remove(src_path)
        except Exception:
            pass
        try:
            if wav_path and os.path.exists(wav_path) and wav_path != src_path:
                os.remove(wav_path)
        except Exception:
            pass

# ---------- DeepSeek ----------
class SummarizeIn(BaseModel):
    transcript: str
    out_lang: str = "zh"
    model: str = "deepseek-chat"

@app.post("/summarize")
def summarize(payload: SummarizeIn):
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY 未配置")

    transcript = (payload.transcript or "").strip()
    if not transcript:
        return {"summary": "", "outline_md": "# 空\n- 无内容"}

    sys_prompt = f"""You are a meeting scribe.
Return STRICT JSON with exactly TWO fields: "summary" and "outline_md".
- "summary": a concise meeting summary in {payload.out_lang} (Markdown allowed).
- "outline_md": a Markdown outline ONLY for markmap (headings/bullets). DO NOT wrap with code fences. DO NOT include any explanation, prefix, or suffix.
No extra keys. No backticks. No prose outside JSON.
"""

    user_prompt = f"""Transcript (language may vary):
\"\"\"{transcript}\"\"\"

Instructions:
1) Write "summary" in {payload.out_lang}.
2) Produce "outline_md" as pure Markdown for a mindmap:
   - Start with a single H1 title.
   - Then use H2/H3 plus bullet lists.
   - No code fences. No triple backticks. No labels like "Outline".
3) Keep it factual and well-structured."""

    try:
        resp = requests.post(
            DEEPSEEK_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": payload.model,
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "response_format": {"type": "text"},
            },
            timeout=60,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"DeepSeek 请求失败: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    content = (
        resp.json()
        .get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        or ""
    )

    data = _extract_json_block(content)
    summary = ""
    outline_md = ""

    if data:
        summary = (data.get("summary") or "").strip()
        outline_md = (data.get("outline_md") or "").strip()

    outline_md = _strip_code_fences(outline_md)

    if not outline_md:
        code_md = _strip_code_fences(content)
        if code_md and len(code_md) > 10 and "# " in code_md:
            outline_md = code_md

    if not outline_md:
        outline_md = _autogen_outline_from_text(summary or transcript, payload.out_lang)

    if outline_md.strip() == "Outline":
        outline_md = _autogen_outline_from_text(summary or transcript, payload.out_lang)

    if summary and not summary.lstrip().startswith("#"):
        summary = f"### 会议摘要\n{summary}"

    return {"summary": summary, "outline_md": outline_md}

# ---------- DeepSeek 连通性自检 ----------
@app.get("/diagnose/deepseek")
def diagnose_deepseek(model: str = "deepseek-chat"):
    if not DEEPSEEK_API_KEY:
        return {"ok": False, "reason": "NO_API_KEY"}

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Reply exactly with: OK"}
        ],
        "temperature": 0.0,
        "response_format": {"type": "text"},
    }
    try:
        r = requests.post(
            DEEPSEEK_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        content = ""
        try:
            content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
        except Exception:
            content = r.text
        return {
            "ok": r.status_code == 200 and "OK" in (content or "").strip(),
            "status_code": r.status_code,
            "content": (content or "")[:200],
        }
    except requests.RequestException as e:
        return {"ok": False, "reason": f"NETWORK_ERROR: {e.__class__.__name__}: {e}"}
