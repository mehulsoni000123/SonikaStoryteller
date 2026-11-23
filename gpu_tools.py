"""Graphics-card helpers for Sonika Storyteller (Windows only).

- boost_this_process(): stop Windows from treating the hidden voice engine as a
  low-priority background job (EcoQoS / power-saving cores). Measured 4x faster.
- scan(): which apps hold memory on the NVIDIA card right now, and how busy they keep it.
- close(): close an app the user picked, gently first, forced only when asked.
"""
import ctypes
import json
import os
import subprocess
import time
from ctypes import wintypes

# Apps that Windows (or this app) needs. They are shown but can never be closed from here.
PROTECTED = {
    "system", "idle", "registry", "smss", "csrss", "wininit", "winlogon", "services", "lsass",
    "svchost", "dwm", "explorer", "fontdrvhost", "sihost", "ctfmon", "textinputhost",
    "startmenuexperiencehost", "shellexperiencehost", "searchhost", "searchapp", "runtimebroker",
    "audiodg", "taskhostw", "lockapp", "applicationframehost", "nvcontainer", "nvdisplay.container",
    "msedgewebview2", "claude", "chrome", "python", "pythonw", "conhost", "cmd", "wscript",
    "securityhealthsystray", "widgets", "phoneexperiencehost",
}
REASON = {
    "dwm": "Draws your screen. Windows needs it",
    "explorer": "Windows taskbar and folders",
    "chrome": "Storyteller's own window runs in Chrome",
    "claude": "The Claude app",
    "msedgewebview2": "Used by Claude and other apps",
    "python": "Background tool or this app's voice engine",
    "pythonw": "Background tool",
}

NICE = {"dwm": "Windows display", "csrss": "Windows core", "explorer": "Windows Explorer",
        "tabtip": "Touch keyboard", "textinputhost": "Windows typing", "searchhost": "Windows Search"}

# Mehul's work apps: only a gentle close (they ask to save). Never force-closed from here.
CAREFUL = {
    "adobe premiere pro", "premiere pro", "afterfx", "aftereffects", "adobe media encoder", "photoshop",
    "lightroom", "illustrator", "cinema 4d", "c4d", "capcut", "comfyui", "handbrake", "resolve",
    "videoproc converter ai", "insta360 studio",
}

NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW
SYS32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
TASKKILL = os.path.join(SYS32, "taskkill.exe")
POWERSHELL = os.path.join(SYS32, "WindowsPowerShell", "v1.0", "powershell.exe")
SCAN_VALID_SECONDS = 180

# pid -> (process name, start time) from the most recent scan, so a recycled pid is never closed by mistake
_last_scan = {"t": 0.0, "pids": {}}
_tried_gentle = set()  # pids that got a gentle close first (force is only allowed after that)


def boost_this_process():
    """Opt this process out of Windows power throttling and raise its priority a little."""
    if os.name != "nt":
        return "not windows"

    class PPT(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.SetProcessInformation.restype = wintypes.BOOL
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.SetPriorityClass.restype = wintypes.BOOL
    h = k32.GetCurrentProcess()
    info = PPT(1, 0x1 | 0x4, 0)  # EXECUTION_SPEED | IGNORE_TIMER_RESOLUTION, state 0 = never throttle
    ok1 = bool(k32.SetProcessInformation(h, 4, ctypes.byref(info), ctypes.sizeof(info)))  # ProcessPowerThrottling
    ok2 = bool(k32.SetPriorityClass(h, 0x8000))  # ABOVE_NORMAL_PRIORITY_CLASS
    return f"power-throttling off={ok1}, priority above-normal={ok2}"


_PS_SCAN = r"""
$ErrorActionPreference = 'SilentlyContinue'
$ad = (Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage').CounterSamples | Sort-Object CookedValue -Descending | Select-Object -First 1
$luid = ($ad.InstanceName -replace '_phys.*','')
$mem = @{}
foreach ($s in (Get-Counter '\GPU Process Memory(*)\Dedicated Usage').CounterSamples) {
  if ($s.InstanceName -like "*$luid*" -and $s.InstanceName -match 'pid_(\d+)_') { $mem[[int]$matches[1]] += $s.CookedValue }
}
$use = @{}
foreach ($s in (Get-Counter '\GPU Engine(*)\Utilization Percentage').CounterSamples) {
  if ($s.InstanceName -like "*$luid*" -and $s.InstanceName -match 'pid_(\d+)_') { $use[[int]$matches[1]] += $s.CookedValue }
}
$out = foreach ($id in $mem.Keys) {
  $mb = [math]::Round($mem[$id] / 1MB)
  $u = [math]::Round([double]$use[$id])
  if ($mb -lt 40 -and $u -lt 3) { continue }
  $p = Get-Process -Id $id
  if (-not $p) { continue }
  $st = 0; try { $st = $p.StartTime.ToFileTimeUtc() } catch {}
  [pscustomobject]@{ pid = $id; start = [string]$st; name = $p.ProcessName; label = $(if ($p.Description) { $p.Description } else { $p.ProcessName }); mb = $mb; busy = [math]::Min(100, $u); window = [bool]$p.MainWindowTitle }
}
@{ adapterMB = [math]::Round($ad.CookedValue / 1MB); apps = @($out) } | ConvertTo-Json -Depth 4 -Compress
"""


def _nvidia_total_mb():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used,utilization.gpu,name",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8,
                           creationflags=NO_WINDOW)
        total, used, util, name = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return int(total), int(used), int(util), name
    except Exception:  # noqa: BLE001
        return None, None, None, "Graphics card"


def scan(self_pids):
    """List apps on the graphics card, grouped by app, biggest first."""
    r = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", _PS_SCAN],
                       capture_output=True, text=True, timeout=40, creationflags=NO_WINDOW)
    try:
        data = json.loads(r.stdout.strip() or "{}")
    except json.JSONDecodeError:
        data = {}
    apps = data.get("apps") or []
    if isinstance(apps, dict):
        apps = [apps]
    groups = {}
    for a in apps:
        key = a["name"].lower()
        g = groups.setdefault(key, {"name": a["name"], "label": a["label"], "mb": 0, "busy": 0, "pids": [],
                                    "starts": {}, "window": False})
        g["starts"][int(a["pid"])] = str(a.get("start") or "0")
        g["mb"] += int(a["mb"])
        g["busy"] = min(100, g["busy"] + int(a["busy"]))
        g["pids"].append(int(a["pid"]))
        g["window"] = g["window"] or bool(a["window"])
    out = []
    for key, g in groups.items():
        is_self = any(p in self_pids for p in g["pids"])
        g["self"] = is_self
        g["protected"] = is_self or key in PROTECTED
        g["careful"] = key in CAREFUL or any(c in (g["label"] or "").lower() for c in CAREFUL)
        g["why"] = "Your voice (this app)" if is_self else REASON.get(key, "Windows needs this" if key in PROTECTED else "")
        if is_self:
            g["label"] = "Sonika Storyteller voice engine"
        elif key in NICE:
            g["label"] = NICE[key]
        out.append(g)
    out.sort(key=lambda g: (-g["mb"], -g["busy"]))
    _last_scan["t"] = time.time()
    _last_scan["pids"] = {p: (g["name"].lower(), g["starts"][p], g["careful"])
                          for g in out if not g["protected"] for p in g["pids"]}
    for g in out:
        g.pop("starts", None)
    total, used, util, card = _nvidia_total_mb()
    return {"card": card, "total_mb": total, "used_mb": used if used is not None else data.get("adapterMB"),
            "util": util, "apps": out}


def _alive(pid):
    r = subprocess.run([os.path.join(SYS32, "tasklist.exe"), "/FI", f"PID eq {int(pid)}", "/NH", "/FO", "CSV"], capture_output=True, text=True,
                       timeout=10, creationflags=NO_WINDOW)
    return f'"{pid}"' in r.stdout


def _identity(pids):
    """pid -> (name, start time) right now, to make sure a pid still belongs to the same app."""
    ids = ",".join(str(int(p)) for p in pids)
    cmd = (f"Get-Process -Id {ids} -EA SilentlyContinue | ForEach-Object {{ $st=0; try {{ $st=$_.StartTime.ToFileTimeUtc() }} catch {{}}; "
           "'{0}|{1}|{2}' -f $_.Id,$_.ProcessName,$st }")
    r = subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", cmd], capture_output=True, text=True,
                       timeout=15, creationflags=NO_WINDOW)
    out = {}
    for line in r.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 3 and parts[0].isdigit():
            out[int(parts[0])] = (parts[1].lower(), parts[2])
    return out


def close(pids, force=False):
    """Close only apps from the latest scan that are not protected, and only if each pid is still the same app."""
    if time.time() - _last_scan["t"] > SCAN_VALID_SECONDS:
        return {"ok": False, "error": "The list is out of date. Press Check again first."}
    pids = [int(p) for p in pids]
    if not pids or len(pids) > 32 or any(p not in _last_scan["pids"] for p in pids):
        return {"ok": False, "error": "That app can't be closed from here."}
    if any(_last_scan["pids"][p][0] in PROTECTED for p in pids):
        return {"ok": False, "error": "Windows needs that app."}
    if force and any(_last_scan["pids"][p][2] for p in pids):
        return {"ok": False, "error": "Your editing apps are never force-closed. Save your work in it, then close it yourself."}
    if force and any(p not in _tried_gentle for p in pids):
        return {"ok": False, "error": "Try a normal close first."}
    now = _identity(pids)
    pids = [p for p in pids if now.get(p) == _last_scan["pids"][p][:2]]  # skip pids that now belong to another app
    if not pids:
        return {"ok": True, "closed": True, "still_open": []}  # the app already closed by itself
    for p in pids:
        if not force:
            _tried_gentle.add(p)
        args = [TASKKILL, "/PID", str(p)] + (["/F"] if force else [])
        subprocess.run(args, capture_output=True, text=True, timeout=15, creationflags=NO_WINDOW)
    # Give apps a moment: a gentle close may show a "save changes?" box first.
    deadline = time.time() + (3 if force else 6)
    left = pids
    while time.time() < deadline:
        left = [p for p in pids if _alive(p)]
        if not left:
            break
        time.sleep(0.5)
    careful = any(_last_scan["pids"][p][2] for p in pids)
    return {"ok": True, "closed": not left, "still_open": left, "careful": careful}
