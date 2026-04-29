# Voice Assistant — Real-Time Streaming Pipeline

A production-grade voice assistant built entirely in Python for Windows 11.
Speak into your microphone — the assistant listens, understands, thinks, and talks back.
Everything runs locally or through standard cloud APIs with no vendor lock-in.

---

## What it does

The assistant implements a full real-time audio pipeline across three engineering phases:

**Phase 1 — Streaming pipeline**
Audio flows from microphone through five async stages without any stage waiting for the previous one to complete:

```
Microphone  ->  VAD  ->  Vosk ASR  ->  Gemini LLM  ->  AWS Polly  ->  Speaker
```

Each stage runs as an independent asyncio coroutine connected by bounded queues.
The result is a continuous, low-latency conversation loop.

**Phase 2 — Latency tracking and dashboard**
Every stage boundary is timestamped with nanosecond precision.
After each utterance, four latency metrics are computed and pushed live to a browser dashboard:
- `ASR latency` — time from first audio chunk to final transcript
- `LLM TTFT` — time from transcript to first token from Gemini
- `TTS TTFB` — time from first token to first audio byte from Polly
- `Total` — microphone to speaker, end to end

The dashboard shows live stacked bars, rolling averages, and a p95 latency history.
A Prometheus-compatible metrics endpoint runs at `http://localhost:9090/metrics`.

**Phase 3 — Resilience and production features**
The system handles failures gracefully without crashing or going silent:
- Every API call has a hard timeout (ASR: 60s, LLM: 30s, TTS: 15s)
- Failed stages restart automatically with exponential backoff
- Pre-recorded fallback audio plays if synthesis fails
- Conversation memory persists across sessions and auto-summarises when long
- Configuration hot-reloads from `config_overrides.json` without restart
- Session audio can be recorded and replayed for debugging

---

## Architecture

```
main.py
  |
  |-- audio_source()    PyAudio mic -> audio_q (500)
  |-- vad_stage()       Silence filter -> vad_q (500)
  |-- asr_stage()       Vosk recognition -> transcript_q (50)
  |-- llm_stage()       Gemini generation -> token_q (200)
  |-- tts_stage()       Polly synthesis -> speaker
       |
       +-- push to dashboard WebSocket (ws://localhost:8765)
       +-- push to metrics collector (http://localhost:9090)
       +-- save to latency_log.json
```

All five stages run concurrently on a single asyncio event loop.
The `Windows SelectorEventLoop` is set explicitly for WebSocket compatibility.

---

## Technology choices

| Component | Technology | Reason |
|-----------|-----------|--------|
| ASR | Vosk (local) | Zero latency, offline, no API cost |
| LLM | Gemini 2.0 Flash | Fastest Gemini model, good quality |
| TTS | AWS Polly Neural | Natural voice, PCM output, low TTFB |
| Audio | PyAudio (16kHz mono PCM) | Direct hardware access, minimal overhead |
| VAD | Energy-based RMS | No extra deps; webrtcvad optional upgrade |
| Queues | asyncio.Queue (bounded) | Back-pressure, no OOM on slow stages |

---

## Quick start

**Prerequisites:**
- Windows 11, Python 3.11 or 3.12
- Working microphone and speakers/headphones

**1. Install**
```bat
install.bat
```

**2. Download Vosk model** (one-time, ~40 MB)
```powershell
Invoke-WebRequest https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip -OutFile model.zip
Expand-Archive model.zip .
Rename-Item vosk-model-small-en-us-0.15 vosk_model
```

**3. Add credentials to `.env`**
```
GEMINI_API_KEY=your_key_here
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_key_here
AWS_REGION=us-east-1
```

**4. Verify**
```bat
python check_setup.py
```

**5. Run**
```bat
python main.py
```
Then open `dashboard.html` in your browser to see live latency metrics.

---

## File reference

```
main.py               Pipeline orchestrator, dashboard server, metrics server
config.py             All parameters (chunk_ms, model names, timeouts, etc.)
mic.py                PyAudio microphone -> async PCM chunk generator
vad.py                Voice activity detection, silence filter, pre-roll buffer
asr.py                Vosk local speech recognition
llm.py                Gemini 2.0 Flash with conversation memory
tts.py                AWS Polly PCM synthesis and playback
latency.py            High-resolution per-utterance timestamp tracker
memory.py             Sliding-window conversation history, auto-summarisation
resilience.py         Timeout, exponential retry, circuit breaker wrappers
interrupt.py          Barge-in: user speech stops TTS mid-playback
wake_word.py          Wake-word gating (openWakeWord or energy-based)
hot_config.py         File-watch live config reload (config_overrides.json)
metrics.py            Prometheus-compatible HTTP metrics server
replay.py             Record sessions to disk, replay for debugging
monitor.py            ANSI terminal live monitor (no browser needed)
dashboard_server.py   Standalone WebSocket server (not needed with main.py)
dashboard.html        Browser latency dashboard
logger.py             Windows-safe coloured console logging
tests.py              66 automated tests, no API keys required
check_setup.py        Pre-flight diagnostics tool
generate_canned.py    Pre-generate Polly fallback audio clips
install.bat           One-click Windows setup script
```

---

## Configuration

Edit `config_overrides.json` while the assistant is running.
Changes apply within 2 seconds — no restart needed.

```json
{
  "vad_threshold": 400,
  "polly_voice": "Matthew",
  "llm_timeout_s": 20.0,
  "debug_logging": true
}
```

| Key | Default | Effect |
|-----|---------|--------|
| `vad_threshold` | 300 | RMS energy to detect speech (lower = more sensitive) |
| `polly_voice` | Joanna | AWS Polly voice (Joanna, Matthew, Amy, Brian...) |
| `llm_timeout_s` | 30.0 | Max wait for Gemini response |
| `tts_timeout_s` | 15.0 | Max wait for Polly synthesis |
| `active_window_s` | 8.0 | Seconds to stay active after wake word |
| `debug_logging` | false | Show DEBUG-level log messages |

---

## Latency

Typical end-to-end latency on a modern Windows laptop with good internet:

| Stage | Typical | What affects it |
|-------|---------|----------------|
| Vosk ASR | 80-150ms | Model size, CPU speed |
| Gemini 2.0 Flash | 300-600ms | Network, query complexity |
| AWS Polly | 150-250ms | Network, response length, AWS region |
| **Total** | **600-1000ms** | |

To lower latency further:
- Use a closer AWS region (e.g. `ap-south-1` for India)
- Increase VAD threshold to reduce false triggers
- Shorter system prompt = faster LLM response

---

## Dashboard

The dashboard is served automatically by `main.py` — no separate server needed.

Open `dashboard.html` in any browser while `main.py` is running.
It connects to `ws://localhost:8765` automatically.

What it shows:
- Live stacked latency bar per utterance (ASR / LLM / TTS / overhead)
- Per-component stat cards with colour coding (green/yellow/red)
- Rolling p50 and p95 latency history
- Event log of recent utterances
- Auto-switches to demo mode when no pipeline is running

---

## Testing

```bat
python tests.py              # all 66 tests
python tests.py --phase 1    # streaming pipeline
python tests.py --phase 2    # latency and memory
python tests.py --phase 3    # resilience and replay
python tests.py -v           # verbose output
```

No API keys required. All external calls are mocked or tested via error paths.

---

## Troubleshooting

**No speech detected (VAD never triggers)**
Add to `config_overrides.json`: `{"vad_threshold": 100}`
Or run without VAD: `python main.py --no-vad`

**Vosk model not found**
Extract the model so `vosk_model\conf\`, `vosk_model\am\`, `vosk_model\graph\` all exist.

**LLM not responding**
Check `GEMINI_API_KEY` in `.env`. Run `python check_setup.py` to diagnose.

**Dashboard shows no data**
Make sure `main.py` is running before opening `dashboard.html`.
Check the browser console for WebSocket errors.
Windows Firewall may block port 8765 — allow Python through firewall.

**Polly not playing audio**
Check AWS credentials. Verify `AmazonPollyReadOnlyAccess` IAM policy is attached.
Run `python generate_canned.py` to pre-generate fallback clips.
