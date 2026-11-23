# Sonika Storyteller

Reads your script aloud in **your own cloned voice**, fully offline on your PC, using the open-source [Chatterbox](https://github.com/resemble-ai/chatterbox) text-to-speech model. English and Hindi lines are detected automatically. Nothing is uploaded anywhere: the server only listens on `127.0.0.1`.

This is a developer-style app: it needs Python and (for good speed) an NVIDIA graphics card. There is no installer.

## Set up (Windows)

1. Install Python 3.11 and create an environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install chatterbox-tts fastapi uvicorn soundfile numpy
   ```
   (Install the CUDA build of PyTorch first if you have an NVIDIA card. See pytorch.org.)
2. Put your own voice recording in `voices\Mehul.wav` (10 to 30 seconds of clean speech). The app shows it as "My voice". For Hindi lines, add `Mehul Hindi.wav` the same way.
3. Set `SONIKA_PYTHON` to the `python.exe` inside your environment (for example `setx SONIKA_PYTHON "C:\path\to\venv\Scripts\python.exe"`), then double-click `Sonika Storyteller.vbs`.

Exports are saved to `Videos\Sonika-Storyteller` (change with `SONIKA_EXPORT_DIR`). The port can be changed with `SONIKA_PORT` (default 7861).

Only clone your own voice, or a voice you have clear permission to use.

## Layout

```
server.py     local server, text planning, caching, voice generation
gpu_tools.py  Windows helpers: speed boost and "which apps use my graphics card"
static/       the app window (index.html)
voices/       your voice samples (not published)
```

License: MIT. Chatterbox and its model have their own licenses.
