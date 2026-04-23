"""
CaritaHub YouTube Transcriber — REST API.

Two-path transcription:
  1. YouTube Caption API (fast, works from any IP, requires captions to exist).
  2. yt-dlp + faster-whisper fallback (downloads audio, runs Whisper on CPU).
"""
from fastapi import FastAPI, BackgroundTasks, HTTPException, Header, Depends, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from faster_whisper import WhisperModel
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
)
import yt_dlp

import os
import re
import glob
import json
import time
import uuid
import base64
import shutil
import sqlite3
import logging
import threading
from typing import Optional, Literal

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("DATA_DIR", "/tmp/caritahub-yt")
DB_PATH = os.environ.get("DB_PATH", os.path.join(DATA_DIR, "jobs.db"))
COOKIE_PATH = os.path.join(DATA_DIR, "cookies.txt")
API_KEY = os.environ.get("API_KEY", "").strip()
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", "3600"))
ALLOWED_MODELS = ("base", "small", "medium", "large")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "base")

os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("caritahub.transcriber")

# ─── Cookie setup (optional) ─────────────────────────────────────────────────
# Priority: YOUTUBE_COOKIES_FILE (path to cookies.txt) > YOUTUBE_COOKIES (base64 or raw content)
_COOKIE_FILE: Optional[str] = None
_cookies_file_path = os.environ.get("YOUTUBE_COOKIES_FILE", "").strip()
_cookie_raw = os.environ.get("YOUTUBE_COOKIES", "").strip()

if _cookies_file_path and os.path.isfile(_cookies_file_path):
    _COOKIE_FILE = _cookies_file_path
    log.info("YouTube cookies loaded from file: %s", _cookies_file_path)
elif _cookie_raw:
    try:
        content = base64.b64decode(_cookie_raw).decode()
    except Exception:
        content = _cookie_raw
    with open(COOKIE_PATH, "w") as f:
        f.write(content)
    _COOKIE_FILE = COOKIE_PATH
    log.info("YouTube cookies loaded from env var")

# ─── SQLite job store ────────────────────────────────────────────────────────
_db_lock = threading.Lock()


def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    with _db() as c:
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id    TEXT PRIMARY KEY,
                status    TEXT DEFAULT 'queued',
                progress  TEXT DEFAULT '',
                created   REAL,
                updated   REAL,
                url       TEXT,
                model     TEXT,
                language  TEXT,
                title     TEXT DEFAULT '',
                duration  REAL DEFAULT 0,
                source    TEXT DEFAULT '',
                error     TEXT DEFAULT '',
                result    TEXT DEFAULT ''
            )
            """
        )


_init_db()


def job_create(job_id: str, url: str, model: str, language: str):
    now = time.time()
    with _db_lock, _db() as c:
        c.execute(
            "INSERT INTO jobs (job_id, url, model, language, created, updated) "
            "VALUES (?,?,?,?,?,?)",
            (job_id, url, model, language, now, now),
        )


def job_update(job_id: str, **kwargs):
    kwargs["updated"] = time.time()
    if "result" in kwargs and not isinstance(kwargs["result"], str):
        kwargs["result"] = json.dumps(kwargs["result"])
    cols = ", ".join(f"{k}=?" for k in kwargs)
    vals = list(kwargs.values()) + [job_id]
    with _db_lock, _db() as c:
        c.execute(f"UPDATE jobs SET {cols} WHERE job_id=?", vals)


def job_get(job_id: str) -> Optional[dict]:
    with _db() as c:
        row = c.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    if d.get("result"):
        try:
            d["result"] = json.loads(d["result"])
        except Exception:
            pass
    return d


def job_delete(job_id: str) -> bool:
    with _db_lock, _db() as c:
        cur = c.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
        removed = cur.rowcount > 0
    shutil.rmtree(os.path.join(DATA_DIR, job_id), ignore_errors=True)
    return removed


_cleanup_lock = threading.Lock()


def cleanup_old_jobs():
    with _cleanup_lock:
        cutoff = time.time() - JOB_TTL_SECONDS
        with _db() as c:
            old = c.execute(
                "SELECT job_id FROM jobs WHERE created < ?", (cutoff,)
            ).fetchall()
        for row in old:
            shutil.rmtree(
                os.path.join(DATA_DIR, row["job_id"]), ignore_errors=True
            )
        with _db_lock, _db() as c:
            c.execute("DELETE FROM jobs WHERE created < ?", (cutoff,))


# ─── Whisper model cache ─────────────────────────────────────────────────────
_model_cache: dict[str, WhisperModel] = {}
_model_lock = threading.Lock()


def get_model(size: str) -> WhisperModel:
    with _model_lock:
        if size not in _model_cache:
            log.info("loading whisper model: %s", size)
            _model_cache[size] = WhisperModel(
                size, device="cpu", compute_type="int8"
            )
        return _model_cache[size]


# ─── Helpers ─────────────────────────────────────────────────────────────────
def format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def extract_video_id(url: str) -> Optional[str]:
    m = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


def segments_to_srt(segments: list[dict]) -> str:
    lines = []
    for i, s in enumerate(segments, 1):
        lines.append(
            f"{i}\n{format_srt_time(s['start'])} --> {format_srt_time(s['end'])}\n{s['text']}\n"
        )
    return "\n".join(lines)


def write_output_files(out_dir: str, transcript: str, srt: str):
    with open(os.path.join(out_dir, "transcript.txt"), "w") as f:
        f.write(transcript)
    with open(os.path.join(out_dir, "subtitles.srt"), "w") as f:
        f.write(srt)


def ydl_meta(url: str) -> dict:
    opts = {
        "skip_download": True,
        "quiet": True,
    }
    if _COOKIE_FILE:
        opts["cookiefile"] = _COOKIE_FILE
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=False)


def ydl_download(url: str, out_dir: str, player_client: Optional[str] = None) -> dict:
    # YouTube on cloud IPs sometimes only serves m3u8 streams (SABR-affected,
    # return empty bytes) + format 18 (direct https mp4 360p w/ audio).
    # Strategy: prefer direct-HTTP audio-only, then ANY direct-HTTP format,
    # then fall back to anything (incl. m3u8). ffmpeg/av extracts audio from mp4 fine.
    opts = {
        "format": (
            "bestaudio[protocol^=http][ext=m4a]/"
            "bestaudio[protocol^=http][ext=webm]/"
            "bestaudio[protocol!=m3u8_native]/"
            "best[protocol^=http]/"
            "bestaudio/best"
        ),
        "outtmpl": f"{out_dir}/audio.%(ext)s",
        "quiet": True,
    }
    if player_client:
        opts["extractor_args"] = {"youtube": {"player_client": [player_client]}}
    if _COOKIE_FILE:
        opts["cookiefile"] = _COOKIE_FILE
    with yt_dlp.YoutubeDL(opts) as ydl:
        return ydl.extract_info(url, download=True)


# ─── Transcription worker ────────────────────────────────────────────────────
def run_transcription(job_id: str, url: str, language: str, model_size: str):
    out_dir = os.path.join(DATA_DIR, job_id)
    os.makedirs(out_dir, exist_ok=True)
    try:
        video_id = extract_video_id(url)

        # Path 1: YouTube caption API
        if video_id:
            try:
                job_update(job_id, status="fetching_captions", progress="fetching captions")
                # youtube-transcript-api v1.x: use instance methods instead of classmethods
                _ytt = YouTubeTranscriptApi()
                tlist = _ytt.list(video_id)
                lang_prefs = (
                    ["en"] if language == "auto" else [language]
                ) + ["en", "zh-Hans", "zh-Hant", "yue", "ja", "ko", "ms"]
                try:
                    t = tlist.find_manually_created_transcript(lang_prefs)
                except Exception:
                    t = tlist.find_generated_transcript(lang_prefs)
                # In v1.x, fetch() returns a FetchedTranscript of snippet objects.
                # In v0.x, it returns list[dict]. Normalize both to list[dict].
                fetched = t.fetch()
                entries = []
                for item in fetched:
                    if isinstance(item, dict):
                        entries.append(item)
                    else:
                        entries.append({
                            "start": getattr(item, "start", 0),
                            "duration": getattr(item, "duration", 0),
                            "text": getattr(item, "text", ""),
                        })

                title, duration = "Unknown", 0
                try:
                    meta = ydl_meta(url)
                    title = meta.get("title", "Unknown")
                    duration = meta.get("duration", 0) or 0
                except Exception as e:
                    log.info("metadata lookup failed: %s", e)

                segments = [
                    {
                        "start": e["start"],
                        "end": e["start"] + e.get("duration", 0),
                        "text": e["text"].strip(),
                    }
                    for e in entries
                ]
                transcript = " ".join(s["text"] for s in segments)
                srt = segments_to_srt(segments)
                write_output_files(out_dir, transcript, srt)

                result = {
                    "title": title,
                    "duration": duration,
                    "language": t.language_code,
                    "source": "youtube_captions",
                    "transcript": transcript,
                    "srt": srt,
                    "segments": segments,
                    "segment_count": len(segments),
                }
                job_update(
                    job_id,
                    status="complete",
                    title=title,
                    duration=duration,
                    language=t.language_code,
                    source="youtube_captions",
                    result=result,
                )
                cleanup_old_jobs()
                return
            except (NoTranscriptFound, TranscriptsDisabled):
                log.info("no captions for %s, falling back to whisper", video_id)
            except Exception as e:
                log.info("caption path failed (%s), falling back to whisper", e)

        # Path 2: yt-dlp + faster-whisper
        # Try default client auto-selection first, then clients known to bypass
        # SABR streaming (tv_embedded, web_embedded), then older fallbacks.
        job_update(job_id, status="downloading", progress="downloading audio")
        errors = []
        info = None
        for client in (None, "tv_embedded", "web_embedded", "ios", "android", "mweb"):
            try:
                info = ydl_download(url, out_dir, player_client=client)
                # Verify a non-empty file was produced before accepting
                import glob as _glob
                audio_files = [
                    f for f in _glob.glob(os.path.join(out_dir, "audio.*"))
                    if not f.endswith(".part") and not f.endswith(".ytdl")
                    and os.path.getsize(f) > 0
                ]
                if not audio_files:
                    # Empty download — SABR or similar, try next client
                    for stale in _glob.glob(os.path.join(out_dir, "audio.*")):
                        try: os.remove(stale)
                        except Exception: pass
                    raise RuntimeError("download produced empty file")
                break
            except Exception as e:
                errors.append(f"{client or 'default'}: {e}")
                info = None
        if info is None:
            raise RuntimeError("yt-dlp failed on all clients: " + " | ".join(errors))

        title = info.get("title", "Unknown")
        duration = info.get("duration", 0) or 0
        job_update(job_id, status="transcribing", title=title, duration=duration)

        audio_files = [
            f
            for f in glob.glob(os.path.join(out_dir, "audio.*"))
            if not f.endswith(".part") and not f.endswith(".ytdl")
        ]
        if not audio_files:
            raise RuntimeError("audio file not found after download")
        audio_file = audio_files[0]

        model = get_model(model_size)
        lang = None if language == "auto" else language
        segments_gen, whisper_info = model.transcribe(
            audio_file, language=lang, word_timestamps=True
        )

        segments, full_text = [], []
        for seg in segments_gen:
            segments.append(
                {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
            )
            full_text.append(seg.text.strip())
            if duration:
                job_update(job_id, progress=f"{int(seg.end)}s / {int(duration)}s")

        transcript = " ".join(full_text)
        srt = segments_to_srt(segments)
        write_output_files(out_dir, transcript, srt)

        detected_lang = (
            whisper_info.language if hasattr(whisper_info, "language") else "unknown"
        )
        result = {
            "title": title,
            "duration": duration,
            "language": detected_lang,
            "source": "whisper",
            "transcript": transcript,
            "srt": srt,
            "segments": segments,
            "segment_count": len(segments),
        }
        job_update(
            job_id,
            status="complete",
            language=detected_lang,
            source="whisper",
            result=result,
        )

        try:
            os.remove(audio_file)
        except Exception:
            pass
        cleanup_old_jobs()
    except Exception as e:
        log.exception("job %s failed", job_id)
        job_update(job_id, status="failed", error=str(e))


# ─── Auth ────────────────────────────────────────────────────────────────────
def require_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing api key")


# ─── Schemas ─────────────────────────────────────────────────────────────────
class TranscribeRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL")
    language: str = Field(
        "auto",
        description="ISO language code (e.g. 'en', 'ja') or 'auto' for autodetect",
    )
    model: Literal["base", "small", "medium", "large"] = Field(
        DEFAULT_MODEL, description="Whisper model size (used only for Path 2 fallback)"
    )


class TranscribeResponse(BaseModel):
    job_id: str
    status: str


class Segment(BaseModel):
    start: float
    end: float
    text: str


class JobResult(BaseModel):
    title: Optional[str] = None
    duration: Optional[float] = None
    language: Optional[str] = None
    source: Optional[str] = None
    transcript: Optional[str] = None
    srt: Optional[str] = None
    segments: Optional[list[Segment]] = None
    segment_count: Optional[int] = None


class JobResponse(BaseModel):
    job_id: str
    status: str
    progress: Optional[str] = None
    title: Optional[str] = None
    duration: Optional[float] = None
    language: Optional[str] = None
    source: Optional[str] = None
    model: Optional[str] = None
    error: Optional[str] = None
    created: Optional[float] = None
    updated: Optional[float] = None
    result: Optional[JobResult] = None


# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="CaritaHub YouTube Transcriber",
    description=(
        "REST API for transcribing YouTube videos. "
        "Uses YouTube captions when available, falls back to Whisper."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "caritahub_youtube_transcriber",
        "version": "1.0.0",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "endpoints": {
            "GET  /api/v1/health": "health check",
            "POST /api/v1/transcribe": "create a transcription job",
            "GET  /api/v1/transcribe/{job_id}": "get job status and result",
            "GET  /api/v1/transcribe/{job_id}/download?format=srt|txt": "download subtitle or transcript file",
            "DELETE /api/v1/transcribe/{job_id}": "delete a job and its files",
        },
    }


@app.get("/api/v1/health", tags=["meta"])
def health():
    return {
        "status": "ok",
        "service": "caritahub_youtube_transcriber",
        "cookies_loaded": bool(_COOKIE_FILE),
        "models_cached": list(_model_cache.keys()),
    }


@app.post(
    "/api/v1/transcribe",
    response_model=TranscribeResponse,
    status_code=202,
    dependencies=[Depends(require_api_key)],
    tags=["transcribe"],
)
def create_transcription(req: TranscribeRequest, background: BackgroundTasks):
    if not extract_video_id(req.url):
        raise HTTPException(400, "url does not contain a recognizable YouTube video id")
    if req.model not in ALLOWED_MODELS:
        raise HTTPException(400, f"model must be one of {list(ALLOWED_MODELS)}")

    job_id = uuid.uuid4().hex
    job_create(job_id, req.url, req.model, req.language)
    background.add_task(run_transcription, job_id, req.url, req.language, req.model)
    log.info("job %s queued for %s (model=%s, lang=%s)", job_id, req.url, req.model, req.language)
    return TranscribeResponse(job_id=job_id, status="queued")


@app.get(
    "/api/v1/transcribe/{job_id}",
    response_model=JobResponse,
    dependencies=[Depends(require_api_key)],
    tags=["transcribe"],
)
def get_transcription(job_id: str):
    job = job_get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return JobResponse(
        job_id=job["job_id"],
        status=job["status"],
        progress=job.get("progress") or None,
        title=job.get("title") or None,
        duration=job.get("duration") or None,
        language=job.get("language") or None,
        source=job.get("source") or None,
        model=job.get("model") or None,
        error=job.get("error") or None,
        created=job.get("created"),
        updated=job.get("updated"),
        result=job.get("result") if isinstance(job.get("result"), dict) else None,
    )


@app.get(
    "/api/v1/transcribe/{job_id}/download",
    dependencies=[Depends(require_api_key)],
    tags=["transcribe"],
)
def download_transcription(
    job_id: str,
    format: Literal["srt", "txt"] = Query(..., description="File format to download"),
):
    job = job_get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job["status"] != "complete":
        raise HTTPException(409, f"job status is '{job['status']}', not complete")

    fname = "subtitles.srt" if format == "srt" else "transcript.txt"
    path = os.path.join(DATA_DIR, job_id, fname)
    if not os.path.exists(path):
        raise HTTPException(404, f"{fname} not found on disk")

    media_type = "application/x-subrip" if format == "srt" else "text/plain"
    return FileResponse(path, media_type=media_type, filename=fname)


@app.delete(
    "/api/v1/transcribe/{job_id}",
    dependencies=[Depends(require_api_key)],
    tags=["transcribe"],
)
def delete_transcription(job_id: str):
    if not job_delete(job_id):
        raise HTTPException(404, "job not found")
    return {"job_id": job_id, "deleted": True}
