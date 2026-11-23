"""Sonika Storyteller - local server.

Reads scripts aloud in Mehul's cloned voice using the local Chatterbox TTS.
Runs only on this PC (127.0.0.1). Nothing is uploaded anywhere.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import gpu_tools

APP_DIR = Path(__file__).resolve().parent
STATIC = APP_DIR / "static"
VOICES = APP_DIR / "voices"
DATA = APP_DIR / "data"
CACHE = APP_DIR / "cache"
EXPORT_DIR = Path(os.environ.get("SONIKA_EXPORT_DIR") or (Path.home() / "Videos" / "Sonika-Storyteller"))
PORT = int(os.environ.get("SONIKA_PORT", "7861"))
IDLE_EXIT_SECONDS = 150  # free the GPU when the app window has been closed
for d in (VOICES, DATA, CACHE):
    d.mkdir(parents=True, exist_ok=True)

LOG = DATA / "server.log"


def log(msg):
    line = time.strftime("%H:%M:%S ") + str(msg)
    print(line, flush=True)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------- text helpers
NUM = r"\d[\d,]*(?:\.\d+)?"
UNIT = {"k": "thousand", "m": "million", "b": "billion"}


DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def lang_of(text: str) -> str:
    """'hi' when the sentence is written in Hindi (Devanagari), else 'en'."""
    return "hi" if DEVANAGARI.search(text or "") else "en"


def spoken(text: str) -> str:
    """Make numbers and symbols easier for the voice model to say."""
    t = text
    if lang_of(t) == "hi":
        t = re.sub(r"(" + NUM + r")\s*%", r"\1 प्रतिशत", t)
        t = t.replace("&", " और ").replace("/", " ")
        t = re.sub(r"(\d),(?=\d)", r"\1", t)
        return re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\$(" + NUM + r")\s*([KkMmBb])\b", lambda m: f"{m.group(1)} {UNIT[m.group(2).lower()]} dollars", t)
    t = re.sub(r"\$(" + NUM + r")", lambda m: f"{m.group(1)} dollars", t)
    t = re.sub(r"(" + NUM + r")\s*%", r"\1 percent", t)
    t = re.sub(r"\b(" + NUM + r")([KkMmBb])\b", lambda m: f"{m.group(1)} {UNIT[m.group(2).lower()]}", t)
    t = t.replace("&", " and ").replace("+", " plus ").replace("/", " ")
    t = re.sub(r"(\d),(\d{3})", r"\1\2", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def trim_silence(wav: np.ndarray, sr: int) -> np.ndarray:
    thr = 0.012
    idx = np.where(np.abs(wav) > thr)[0]
    if len(idx) == 0:
        return wav
    pad = int(sr * 0.05)
    start = max(0, idx[0] - pad)
    end = min(len(wav), idx[-1] + int(sr * 0.12))
    return wav[start:end]


# ---------------------------------------------------------------- TTS engine
class Engine:
    def __init__(self):
        self.model = None
        self.kind = None   # "en" (English model) or "mtl" (Hindi + English model)
        self.want = "en"   # which model the current script needs
        self.conds = {}    # voice id -> prepared voice conditionals
        self.state = "loading"  # loading | ready | error
        self.error = ""
        self.device = ""
        self.cond_voice = None
        self.lock = threading.Lock()
        self.plan = []  # list of dict(key, text, exag)
        self.voice = "Mehul"
        self.cfg = 0.5
        self.focus = 0
        self.failed = {}  # key -> attempts
        self.busy_key = None
        self.speed_samples = []  # (audio_seconds, gen_seconds)
        self.wake = threading.Event()
        threading.Thread(target=self._load, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()

    # -- loading
    def _load(self):
        try:
            import gc
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            kind = self.want
            # Only one model fits next to other apps on an 8 GB card: free the old one first.
            if self.model is not None:
                self.model = None
                self.conds = {}
                self.cond_voice = None
                gc.collect()
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            log(f"Loading Chatterbox ({'Hindi + English' if kind == 'mtl' else 'English'}) on {self.device} ...")
            t0 = time.time()
            if kind == "mtl":
                from chatterbox.mtl_tts import ChatterboxMultilingualTTS
                self.model = ChatterboxMultilingualTTS.from_pretrained(device=self.device)
            else:
                from chatterbox.tts import ChatterboxTTS
                self.model = ChatterboxTTS.from_pretrained(device=self.device)
            self.kind = kind
            self._instrument()
            log(f"Model ready in {time.time() - t0:.1f}s")
            self.state = "ready"
            self.wake.set()
        except Exception as e:  # noqa: BLE001
            self.state = "error"
            msg = str(e)
            if "out of memory" in msg.lower():
                msg = "The graphics card is full. Close Premiere, After Effects or ComfyUI, then press Retry."
            self.error = msg
            log("LOAD FAILED: " + traceback.format_exc())

    def _instrument(self):
        """Time the slow parts of generation so the log shows where time goes."""
        self.timing = {"s3gen": 0.0, "watermark": 0.0}
        m = self.model
        for obj, attr, name in ((m.s3gen, "inference", "s3gen"), (m.watermarker, "apply_watermark", "watermark")):
            fn = getattr(obj, attr)

            def wrapped(*a, _fn=fn, _name=name, **k):
                t0 = time.time()
                try:
                    return _fn(*a, **k)
                finally:
                    self.timing[_name] += time.time() - t0
            setattr(obj, attr, wrapped)

    def retry_load(self):
        if self.state == "error":
            self.state = "loading"
            self.error = ""
            threading.Thread(target=self._load, daemon=True).start()

    # -- planning
    @staticmethod
    def voice_for(voice, lang):
        """Hindi lines use the Hindi recording of the same voice (e.g. 'Mehul Hindi')."""
        if lang == "hi":
            for name in (f"{voice} Hindi", "Mehul Hindi"):
                if (VOICES / f"{name}.wav").exists():
                    return name
        return voice

    @classmethod
    def key_for(cls, text, voice, exag, cfg, model="en"):
        lang = lang_of(text)
        voice = cls.voice_for(voice, lang)
        vp = VOICES / f"{voice}.wav"
        stamp = int(vp.stat().st_mtime) if vp.exists() else 0
        tag = "v2" if model == "en" else f"mtl|{lang}"
        raw = f"{tag}|{spoken(text)}|{voice}|{stamp}|{exag:.2f}|{cfg:.2f}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]

    def set_plan(self, items, voice, cfg, focus):
        # Any Hindi line -> the Hindi + English model reads the whole script (no model swapping mid-script).
        want = "mtl" if any(lang_of(it["text"]) == "hi" for it in items) else "en"
        with self.lock:
            self.voice = voice
            self.cfg = cfg
            self.want = want
            self.plan = [
                {"key": self.key_for(it["text"], voice, it["exag"], cfg, want), "text": it["text"],
                 "exag": it["exag"], "lang": lang_of(it["text"])}
                for it in items
            ]
            self.focus = max(0, min(focus, len(self.plan) - 1)) if self.plan else 0
            keys = [p["key"] for p in self.plan]
        self.wake.set()
        return keys

    def set_focus(self, idx):
        with self.lock:
            self.focus = max(0, idx)
        self.wake.set()

    def ready(self, key):
        return (CACHE / f"{key}.wav").exists()

    def _next_job(self):
        with self.lock:
            n = len(self.plan)
            if not n:
                return None
            order = list(range(self.focus, n)) + list(range(0, self.focus))
            for i in order:
                job = self.plan[i]
                if not self.ready(job["key"]) and self.failed.get(job["key"], 0) < 3:
                    return dict(job, voice=self.voice_for(self.voice, job.get("lang", "en")), cfg=self.cfg)
        return None

    # -- generation
    def _voice_path(self, name):
        p = VOICES / f"{name}.wav"
        if not p.exists():
            p = VOICES / "Mehul.wav"
        return p

    def _generate(self, job):
        import torch
        m = self.model
        vp = self._voice_path(job["voice"])
        vid = f"{vp}|{vp.stat().st_mtime}"
        if self.cond_voice != vid:
            if vid not in self.conds:
                m.prepare_conditionals(str(vp), exaggeration=job["exag"])
                self.conds[vid] = m.conds
            m.conds = self.conds[vid]
            self.cond_voice = vid
        text = spoken(job["text"])
        lang = job.get("lang", "en")
        if self.kind == "mtl":
            gen = lambda: m.generate(text, language_id=lang, exaggeration=job["exag"], cfg_weight=job["cfg"])
        else:
            gen = lambda: m.generate(text, exaggeration=job["exag"], cfg_weight=job["cfg"])
        words = max(1, len(text.split()))
        expected = words / 2.6 + 0.6
        best = None
        for attempt in range(2):
            t0 = time.time()
            wav = gen().squeeze().cpu().numpy()
            gen_t = time.time() - t0
            tm = getattr(self, "timing", {})
            log(f"gen {gen_t:.2f}s for {len(wav) / m.sr:.1f}s audio | s3gen {tm.get('s3gen', 0):.2f}s watermark {tm.get('watermark', 0):.2f}s")
            self.timing = {"s3gen": 0.0, "watermark": 0.0}
            wav = trim_silence(wav.astype(np.float32), m.sr)
            dur = len(wav) / m.sr
            self.speed_samples = (self.speed_samples + [(dur, gen_t)])[-12:]
            # Guard against the model rambling or cutting off: retry once.
            if best is None or abs(dur - expected) < abs(len(best) / m.sr - expected):
                best = wav
            if expected * 0.45 <= dur <= expected * 2.2 + 1.5:
                break
            log(f"Odd length ({dur:.1f}s vs ~{expected:.1f}s), retrying: {text[:60]}")
        tmp = CACHE / f"{job['key']}.tmp.wav"
        sf.write(str(tmp), best, m.sr)
        os.replace(tmp, CACHE / f"{job['key']}.wav")
        if self.device == "cuda":
            torch.cuda.empty_cache() if len(self.speed_samples) % 20 == 0 else None

    def _worker(self):
        while True:
            if self.state != "ready":
                self.wake.wait(1.0)
                self.wake.clear()
                continue
            if self.want != self.kind:
                self.state = "loading"
                self._load()
                continue
            job = self._next_job()
            if not job:
                self.wake.wait(2.0)
                self.wake.clear()
                continue
            self.busy_key = job["key"]
            try:
                self._generate(job)
            except Exception as e:  # noqa: BLE001
                self.failed[job["key"]] = self.failed.get(job["key"], 0) + 1
                log(f"Generate failed ({self.failed[job['key']]}x): {e}")
                if "out of memory" in str(e).lower():
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except Exception:  # noqa: BLE001
                        pass
                    self.error = "Graphics card memory is full. Close other GPU apps (Premiere, ComfyUI)."
                    time.sleep(3)
            finally:
                self.busy_key = None

    def eta(self):
        """Seconds until the whole plan is ready (rough)."""
        rtf = self.realtime_factor() or 1.3
        with self.lock:
            words = sum(len(p["text"].split()) for p in self.plan if not self.ready(p["key"]))
        return round(words / 2.6 / rtf + (3 if words else 0))

    def is_slow(self):
        """True when the last few lines took far longer than they last (graphics card busy elsewhere)."""
        recent = self.speed_samples[-3:]
        if not recent:
            return False
        a = sum(x[0] for x in recent)
        g = sum(x[1] for x in recent)
        return g > 0 and a / g < 0.45

    def gpu_free_mb(self):
        try:
            import torch
            if self.device == "cuda":
                return int(torch.cuda.mem_get_info()[0] / 2**20)
        except Exception:  # noqa: BLE001
            pass
        return None

    def realtime_factor(self):
        a = sum(s[0] for s in self.speed_samples)
        g = sum(s[1] for s in self.speed_samples)
        return round(a / g, 2) if g else None


engine = Engine()

# ---------------------------------------------------------------- library storage
LIB = DATA / "library.json"
lib_lock = threading.Lock()


def lib_load():
    if LIB.exists():
        try:
            return json.loads(LIB.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            LIB.rename(DATA / f"library.broken.{int(time.time())}.json")
    return {}


def lib_save(lib):
    tmp = DATA / "library.tmp.json"
    tmp.write_text(json.dumps(lib, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, LIB)


# ---------------------------------------------------------------- API
app = FastAPI()
last_ping = time.time()


class PlanReq(BaseModel):
    items: list
    voice: str = "Mehul"
    cfg: float = 0.5
    focus: int = 0


class FocusReq(BaseModel):
    focus: int


class ScriptReq(BaseModel):
    id: str | None = None
    title: str = "Untitled script"
    text: str = ""
    pos: int = 0
    marks: dict = {}
    settings: dict = {}
    keys: list[str] = []
    project: str | None = None   # only used when the script is first saved


class NameReq(BaseModel):
    name: str


class MoveReq(BaseModel):
    project: str = ""


class ProjectDeleteReq(BaseModel):
    with_parts: bool = False


class ExportReq(BaseModel):
    keys: list
    title: str = "script"
    gap: float = 0.3
    gaps: list | None = None


ALLOWED_ORIGINS = {f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"}


@app.middleware("http")
async def no_cache(request, call_next):
    # Only this app's own window may use the server (blocks other web pages from poking it).
    if request.headers.get("host", "").split(":")[0] not in ("127.0.0.1", "localhost"):
        return JSONResponse({"ok": False, "error": "Not allowed"}, status_code=403)
    if request.method == "POST":
        origin = request.headers.get("origin")
        if request.headers.get("x-sonika") != "1" or (origin and origin not in ALLOWED_ORIGINS):
            return JSONResponse({"ok": False, "error": "Not allowed"}, status_code=403)
    resp = await call_next(request)
    if request.url.path.startswith("/api/") or request.url.path in ("/", "/index.html"):
        resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/api/status")
def status():
    with engine.lock:
        keys = [p["key"] for p in engine.plan]
        focus = engine.focus
    ready = [engine.ready(k) for k in keys]
    ahead = 0
    for r in ready[focus:]:
        if not r:
            break
        ahead += 1
    return {
        "state": engine.state,
        "error": engine.error,
        "device": engine.device,
        "total": len(keys),
        "ready_count": sum(ready),
        "ready": ready,
        "ahead": ahead,
        "busy": engine.busy_key,
        "rtf": engine.realtime_factor(),
        "eta": engine.eta(),
        "slow": engine.is_slow(),
        "gpu_free_mb": engine.gpu_free_mb(),
        "model": engine.kind,
        "want": engine.want,
    }


@app.post("/api/retry")
def retry():
    engine.retry_load()
    engine.failed.clear()
    engine.error = ""
    return {"ok": True}


@app.get("/api/voices")
def voices():
    # "<name> Hindi" recordings are used automatically for Hindi lines, so they are not listed separately.
    names = sorted(p.stem for p in VOICES.glob("*.wav") if not p.stem.endswith(" Hindi"))
    if "Mehul" in names:
        names.remove("Mehul")
        names.insert(0, "Mehul")
    return {"voices": names}


@app.post("/api/plan")
def plan(req: PlanReq):
    items = []
    for it in req.items:
        text = str(it.get("text", "")).strip()
        exag = float(it.get("exag", 0.5))
        items.append({"text": text, "exag": max(0.25, min(1.2, exag))})
    keys = engine.set_plan(items, req.voice, max(0.0, min(1.0, req.cfg)), req.focus)
    return {"keys": keys}


@app.post("/api/keys")
def keys_only(req: PlanReq):
    """Audio file names for a script without starting any voice work (used to measure size)."""
    items = [{"text": str(it.get("text", "")).strip(), "exag": max(0.25, min(1.2, float(it.get("exag", 0.5))))} for it in req.items]
    model = "mtl" if any(lang_of(it["text"]) == "hi" for it in items) else "en"
    cfg = max(0.0, min(1.0, req.cfg))
    return {"keys": [engine.key_for(it["text"], req.voice, it["exag"], cfg, model) for it in items]}


@app.post("/api/library/{sid}/keys")
def library_keys(sid: str, body: dict):
    with lib_lock:
        lib = lib_load()
        if sid not in lib:
            raise HTTPException(404)
        keys = lib[sid].get("keys", [])
        for k in body.get("keys", []):
            if isinstance(k, str) and KEY_RE.fullmatch(k) and k not in keys:
                keys.append(k)
        lib[sid]["keys"] = keys[-5000:]
        lib_save(lib)
    return {"ok": True}


@app.post("/api/focus")
def focus(req: FocusReq):
    engine.set_focus(req.focus)
    return {"ok": True}


@app.get("/api/audio/{key}")
def audio(key: str):
    if not re.fullmatch(r"[0-9a-f]{20}", key):
        raise HTTPException(400)
    p = CACHE / f"{key}.wav"
    if p.exists():
        return FileResponse(p, media_type="audio/wav")
    return Response(status_code=202)


@app.post("/api/ping")
def ping():
    global last_ping
    last_ping = time.time()
    return {"ok": True}


@app.post("/api/quit")
def quit_app():
    threading.Timer(0.5, lambda: os._exit(0)).start()
    return {"ok": True}


PROJECTS = DATA / "projects.json"
KEY_RE = re.compile(r"[0-9a-f]{20}")


def projects_load():
    if PROJECTS.exists():
        try:
            return json.loads(PROJECTS.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            PROJECTS.rename(DATA / f"projects.broken.{int(time.time())}.json")
    return {}


def projects_save(pr):
    tmp = DATA / "projects.tmp.json"
    tmp.write_text(json.dumps(pr, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, PROJECTS)


def audio_bytes(keys):
    total = 0
    for k in keys:
        f = CACHE / f"{k}.wav"
        if f.exists():
            total += f.stat().st_size
    return total


def mb(n):
    return round(n / 2**20, 1)


def delete_script_locked(lib, sid):
    """Move the script to removed.json (text kept as a backup) and delete audio that no other script uses."""
    s = lib.pop(sid, None)
    if not s:
        return 0
    removed_p = DATA / "removed.json"
    removed = json.loads(removed_p.read_text(encoding="utf-8")) if removed_p.exists() else {}
    removed[sid] = dict(s, deleted=time.time())
    removed_p.write_text(json.dumps(removed, ensure_ascii=False, indent=1), encoding="utf-8")
    still_used = {k for other in lib.values() for k in other.get("keys", [])}
    freed = 0
    for k in set(s.get("keys", [])) - still_used:
        f = CACHE / f"{k}.wav"
        if f.exists():
            freed += f.stat().st_size
            f.unlink()
    return freed


@app.get("/api/library")
def library():
    with lib_lock:
        lib = lib_load()
        pr = projects_load()
    items = sorted(lib.values(), key=lambda s: s.get("updated", 0), reverse=True)
    scripts = []
    for s in items:
        d = {k: s.get(k) for k in ("id", "title", "updated", "pos", "count", "created")}
        d["done"] = bool((s.get("settings") or {}).get("done"))
        d["project"] = s.get("project") or ""
        d["mb"] = mb(audio_bytes(s.get("keys", []))) if s.get("keys") else None
        scripts.append(d)
    projects = sorted(pr.values(), key=lambda p: p.get("created", 0), reverse=True)
    all_audio = sum(f.stat().st_size for f in CACHE.glob("*.wav"))
    return {"scripts": scripts, "projects": projects, "audio_mb": mb(all_audio)}


@app.get("/api/library/{sid}")
def library_get(sid: str):
    with lib_lock:
        lib = lib_load()
    if sid not in lib:
        raise HTTPException(404)
    return lib[sid]


@app.post("/api/library")
def library_put(req: ScriptReq):
    with lib_lock:
        lib = lib_load()
        sid = req.id or uuid.uuid4().hex[:10]
        old = lib.get(sid, {})
        keys = [k for k in old.get("keys", []) if KEY_RE.fullmatch(k)]
        for k in req.keys:
            if KEY_RE.fullmatch(k) and k not in keys:
                keys.append(k)
        project = old.get("project", "")
        if not old and req.project and req.project in projects_load():
            project = req.project
        lib[sid] = {
            "id": sid,
            "project": project,
            "keys": keys[-5000:],
            "title": (old.get("title") if old else None) or req.title[:120] or "Untitled script",
            "text": req.text,
            "pos": req.pos,
            "marks": req.marks,
            "settings": req.settings,
            "count": req.settings.get("count", old.get("count", 0)),
            "created": old.get("created", time.time()),
            "updated": time.time(),
        }
        lib_save(lib)
    return {"id": sid}


@app.post("/api/library/{sid}/rename")
def library_rename(sid: str, req: NameReq):
    name = req.name.strip()[:120]
    if not name:
        return {"ok": False, "error": "Type a name first."}
    with lib_lock:
        lib = lib_load()
        if sid not in lib:
            raise HTTPException(404)
        lib[sid]["title"] = name
        lib_save(lib)
    return {"ok": True}


@app.post("/api/library/{sid}/move")
def library_move(sid: str, req: MoveReq):
    with lib_lock:
        lib = lib_load()
        if sid not in lib or (req.project and req.project not in projects_load()):
            raise HTTPException(404)
        lib[sid]["project"] = req.project
        lib[sid]["created"] = time.time()  # goes to the end of the project as the newest part
        lib_save(lib)
    return {"ok": True}


@app.post("/api/library/{sid}/delete")
def library_delete(sid: str):
    with lib_lock:
        lib = lib_load()
        if sid not in lib:
            raise HTTPException(404)
        freed = delete_script_locked(lib, sid)
        lib_save(lib)
    log(f"Deleted script {sid}, freed {mb(freed)} MB")
    return {"ok": True, "freed_mb": mb(freed)}


@app.post("/api/projects")
def project_new(req: NameReq):
    name = req.name.strip()[:80]
    if not name:
        return {"ok": False, "error": "Type a name first."}
    with lib_lock:
        pr = projects_load()
        pid = "p" + uuid.uuid4().hex[:9]
        pr[pid] = {"id": pid, "name": name, "created": time.time()}
        projects_save(pr)
    return {"ok": True, "id": pid}


@app.post("/api/projects/{pid}/rename")
def project_rename(pid: str, req: NameReq):
    name = req.name.strip()[:80]
    if not name:
        return {"ok": False, "error": "Type a name first."}
    with lib_lock:
        pr = projects_load()
        if pid not in pr:
            raise HTTPException(404)
        pr[pid]["name"] = name
        projects_save(pr)
    return {"ok": True}


@app.post("/api/projects/{pid}/delete")
def project_delete(pid: str, req: ProjectDeleteReq):
    with lib_lock:
        pr = projects_load()
        if pid not in pr:
            raise HTTPException(404)
        lib = lib_load()
        freed, parts = 0, [sid for sid, s in lib.items() if s.get("project") == pid]
        for sid in parts:
            if req.with_parts:
                freed += delete_script_locked(lib, sid)
            else:
                lib[sid]["project"] = ""
        lib_save(lib)
        pr.pop(pid)
        projects_save(pr)
    log(f"Deleted project {pid} ({len(parts)} parts, {'deleted' if req.with_parts else 'kept'}), freed {mb(freed)} MB")
    return {"ok": True, "freed_mb": mb(freed)}


@app.post("/api/library/{sid}/hide")
def library_hide(sid: str):
    """Remove from the list (kept in data/removed.json, never deleted)."""
    with lib_lock:
        lib = lib_load()
        if sid in lib:
            removed_p = DATA / "removed.json"
            removed = json.loads(removed_p.read_text(encoding="utf-8")) if removed_p.exists() else {}
            removed[sid] = lib.pop(sid)
            removed_p.write_text(json.dumps(removed, ensure_ascii=False, indent=1), encoding="utf-8")
            lib_save(lib)
    return {"ok": True}


@app.post("/api/export")
def export(req: ExportReq):
    paths = [CACHE / f"{k}.wav" for k in req.keys]
    missing = sum(1 for p in paths if not p.exists())
    if missing:
        return JSONResponse({"ok": False, "error": f"{missing} sentences are still being prepared."}, status_code=409)
    chunks, sr = [], None
    for i, p in enumerate(paths):
        w, sr = sf.read(str(p), dtype="float32")
        chunks.append(w)
        g = req.gaps[i] if req.gaps and i < len(req.gaps) else req.gap
        chunks.append(np.zeros(int(sr * max(0.0, float(g))), dtype=np.float32))
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9 _-]+", "", req.title).strip()[:60] or "script"
    stamp = time.strftime("%Y-%m-%d_%H%M")
    wav_p = CACHE / "export_tmp.wav"
    mp3_p = EXPORT_DIR / f"{safe}_{stamp}.mp3"
    sf.write(str(wav_p), np.concatenate(chunks), sr)
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_p), "-b:a", "160k", str(mp3_p)],
                       capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        return JSONResponse({"ok": False, "error": r.stderr[-300:]}, status_code=500)
    return {"ok": True, "path": str(mp3_p)}


SETTINGS = DATA / "settings.json"


@app.get("/api/settings")
def settings_get():
    try:
        return json.loads(SETTINGS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@app.post("/api/settings")
def settings_put(body: dict):
    cur = settings_get()
    cur.update(body)
    SETTINGS.write_text(json.dumps(cur, indent=1), encoding="utf-8")
    return cur


def best_window(wav: np.ndarray, sr: int, seconds: float = 14.0) -> np.ndarray:
    """Pick the cleanest, most continuous stretch of speech for voice cloning."""
    if len(wav) <= sr * seconds:
        return wav
    hop = int(sr * 0.05)
    frames = np.array([np.sqrt(np.mean(wav[i:i + hop] ** 2)) for i in range(0, len(wav) - hop, hop)])
    speech = frames > max(0.01, np.percentile(frames, 60) * 0.25)
    win = int(seconds / 0.05)
    best, best_score = 0, -1.0
    for start in range(0, len(frames) - win, 4):
        seg = speech[start:start + win]
        # reward speech, punish long silences inside the window
        runs, longest = 0, 0
        for v in seg:
            runs = 0 if v else runs + 1
            longest = max(longest, runs)
        score = seg.mean() - 0.04 * max(0, longest - 12)
        if score > best_score:
            best, best_score = start, score
    a = best * hop
    return wav[a:a + int(sr * seconds)]


@app.post("/api/voices/import")
async def voice_import(request: Request, name: str = "My English voice"):
    data = await request.body()
    if len(data) < 2000:
        return JSONResponse({"ok": False, "error": "That file looks empty."}, status_code=400)
    safe = re.sub(r"[^A-Za-z0-9 _-]+", "", name).strip()[:40] or "My voice"
    src = DATA / "import_src.bin"
    full = DATA / "import_full.wav"
    src.write_bytes(data)
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "24000",
                        "-af", "highpass=f=70,loudnorm=I=-18:TP=-2", str(full)],
                       capture_output=True, text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        return JSONResponse({"ok": False, "error": "Could not read that audio or video file."}, status_code=400)
    wav, sr = sf.read(str(full), dtype="float32")
    if len(wav) < sr * 6:
        return JSONResponse({"ok": False, "error": "Please use at least 10 seconds of clear speech."}, status_code=400)
    clip = best_window(wav, sr)
    sf.write(str(VOICES / f"{safe}.wav"), clip, sr)
    src.unlink(missing_ok=True)
    full.unlink(missing_ok=True)
    log(f"Imported voice '{safe}' ({len(clip) / sr:.1f}s clip from {len(wav) / sr:.0f}s recording)")
    return {"ok": True, "name": safe, "seconds": round(len(clip) / sr, 1)}


@app.post("/api/open-folder")
def open_folder():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    os.startfile(str(EXPORT_DIR))
    return {"ok": True}


class GpuCloseReq(BaseModel):
    pids: list[int]
    force: bool = False
    name: str = ""


@app.get("/api/gpu")
def gpu():
    t0 = time.time()
    data = gpu_tools.scan({os.getpid(), os.getppid()})
    data["voice_speed"] = engine.realtime_factor()
    data["slow"] = engine.is_slow()
    log(f"GPU check in {time.time() - t0:.1f}s: " + ", ".join(f"{a['name']} {a['mb']}MB" for a in data["apps"][:6]))
    return data


@app.post("/api/gpu/close")
def gpu_close(req: GpuCloseReq):
    res = gpu_tools.close(req.pids, req.force)
    log(f"Close {'(forced) ' if req.force else ''}{req.name} {req.pids}: {res}")
    return res


app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")


def watchdog():
    while True:
        time.sleep(15)
        if time.time() - last_ping > IDLE_EXIT_SECONDS:
            log("App window closed for a while - shutting down to free the graphics card.")
            os._exit(0)


if __name__ == "__main__":
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        log("Speed boost: " + gpu_tools.boost_this_process())
    except Exception as e:  # noqa: BLE001
        log(f"Speed boost failed: {e}")
    log(f"Sonika Storyteller on http://127.0.0.1:{PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
