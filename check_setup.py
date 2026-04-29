"""
check_setup.py -- Pre-flight diagnostics for Windows 11.

Run before first use to verify all dependencies, credentials,
Vosk model, and audio hardware are working correctly.

Usage:
    python check_setup.py
    python check_setup.py --audio    include 2-second mic recording test
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Enable ANSI colours on Windows
try:
    import ctypes, ctypes.wintypes
    k = ctypes.windll.kernel32
    h = k.GetStdHandle(-11); m = ctypes.wintypes.DWORD()
    k.GetConsoleMode(h, ctypes.byref(m))
    k.SetConsoleMode(h, m.value | 0x0004)
    k.SetConsoleOutputCP(65001)
    _ANSI = True
except Exception:
    _ANSI = False

def _c(s): return s if _ANSI else ""
OK   = _c("\033[32m") + "[OK]  " + _c("\033[0m")
FAIL = _c("\033[31m") + "[FAIL]" + _c("\033[0m")
WARN = _c("\033[33m") + "[WARN]" + _c("\033[0m")

_results: list[dict] = []

def check(label: str, passed: bool, detail: str = "", warn: bool = False):
    icon = OK if passed else (WARN if warn else FAIL)
    print(f"  {icon}  {label}" + (f"  ({detail})" if detail else ""))
    _results.append({"label": label, "passed": passed, "warn": warn})
    return passed


def check_python():
    print("\n-- Python --")
    v = sys.version_info
    check("Python 3.11+", v >= (3, 11), f"found {v.major}.{v.minor}.{v.micro}")
    check("Windows platform", sys.platform == "win32", sys.platform)


def check_packages():
    print("\n-- Packages --")
    required = [
        ("pyaudio",       "pyaudio"),
        ("vosk",          "vosk"),
        ("websockets",    "websockets"),
        ("google-genai",  "google.genai"),
        ("boto3",         "boto3"),
        ("python-dotenv", "dotenv"),
    ]
    optional = [
        ("webrtcvad-wheels", "webrtcvad", "better VAD accuracy"),
        ("openwakeword",     "openwakeword", "neural wake word"),
    ]
    for pkg, mod in required:
        try:
            __import__(mod); check(pkg, True)
        except ImportError as e:
            check(pkg, False, str(e))
    for pkg, mod, tip in optional:
        try:
            __import__(mod); check(f"{pkg}  [{tip}]", True)
        except ImportError:
            check(f"{pkg}  [{tip}]", False, f"pip install {pkg}", warn=True)


def check_credentials():
    print("\n-- Credentials (.env) --")
    env_path = Path(".env")
    if env_path.exists():
        check(".env file present", True)
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
    else:
        check(".env file present", False, "copy .env.example to .env and fill in keys")

    for var, service in [
        ("GEMINI_API_KEY",       "Gemini LLM"),
        ("AWS_ACCESS_KEY_ID",    "AWS Polly"),
        ("AWS_SECRET_ACCESS_KEY","AWS Polly"),
    ]:
        val = os.environ.get(var, "")
        ok  = bool(val and "your_" not in val.lower())
        check(f"{var}", ok, "set" if ok else "missing or placeholder")


def check_vosk_model():
    print("\n-- Vosk model --")
    model_path = Path(os.environ.get("VOSK_MODEL_PATH", "vosk_model"))
    if not check(f"Model directory  {model_path}", model_path.exists() and model_path.is_dir(),
                 "see README -- download from alphacephei.com/vosk/models"):
        return
    valid = all((model_path / d).exists() for d in ("am", "conf", "graph"))
    if not check("Model structure valid  (am/ conf/ graph/)", valid,
                 "re-extract the model archive"):
        return
    try:
        from vosk import Model, SetLogLevel
        SetLogLevel(-1)
        t0 = time.monotonic()
        Model(str(model_path))
        check("Model loads successfully", True, f"{time.monotonic()-t0:.1f}s load time")
    except Exception as e:
        check("Model loads successfully", False, str(e))


def check_apis():
    print("\n-- API connectivity --")

    def _gemini():
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key: return False, "no key"
        try:
            from google import genai
            client = genai.Client(api_key=key)
            models = list(client.models.list())
            return bool(models), f"{len(models)} models available"
        except Exception as e:
            return False, str(e)

    def _polly():
        key = os.environ.get("AWS_ACCESS_KEY_ID", "")
        sec = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
        reg = os.environ.get("AWS_REGION", "us-east-1")
        if not key: return False, "no credentials"
        try:
            import boto3
            boto3.client("polly", region_name=reg,
                         aws_access_key_id=key,
                         aws_secret_access_key=sec).describe_voices(LanguageCode="en-US")
            return True, f"region={reg}"
        except Exception as e:
            return False, str(e)

    for label, fn in [("Gemini 2.0 Flash", _gemini), ("AWS Polly", _polly)]:
        ok, detail = fn()
        check(label, ok, detail)


def check_audio():
    print("\n-- Audio hardware --")
    try:
        import pyaudio
        pa = pyaudio.PyAudio()
        inputs  = [pa.get_device_info_by_index(i)["name"]
                   for i in range(pa.get_device_count())
                   if pa.get_device_info_by_index(i).get("maxInputChannels", 0) > 0]
        outputs = [pa.get_device_info_by_index(i)["name"]
                   for i in range(pa.get_device_count())
                   if pa.get_device_info_by_index(i).get("maxOutputChannels", 0) > 0]
        pa.terminate()
        check("Microphone", bool(inputs),  inputs[0][:60]  if inputs  else "none found")
        check("Speaker",    bool(outputs), outputs[0][:60] if outputs else "none found")
    except Exception as e:
        check("Audio hardware", False, str(e))


def check_audio_recording():
    print("\n-- Microphone recording test (2 seconds) --")
    print("  Speak now...")
    try:
        import pyaudio, struct, math
        CHUNK = 320   # 20ms at 16kHz
        pa    = pyaudio.PyAudio()
        st    = pa.open(format=pyaudio.paInt16, channels=1, rate=16000,
                        input=True, frames_per_buffer=CHUNK)
        energies = []
        for _ in range(int(16000 / CHUNK * 2)):
            data = st.read(CHUNK, exception_on_overflow=False)
            s    = struct.unpack(f"<{len(data)//2}h", data)
            energies.append(math.sqrt(sum(x*x for x in s) / len(s)))
        st.stop_stream(); st.close(); pa.terminate()
        avg  = sum(energies) / len(energies)
        peak = max(energies)
        check("Audio captured", True, f"avg={avg:.0f} RMS  peak={peak:.0f} RMS")
        if avg < 50:
            print(f"  {WARN}  Very quiet -- lower vad_threshold in config_overrides.json")
        elif avg > 8000:
            print(f"  {WARN}  Very loud -- consider lowering mic gain in Windows settings")
        else:
            print(f"  {OK}  Audio level looks good for VAD")
    except Exception as e:
        check("Audio capture", False, str(e))


def main():
    p = argparse.ArgumentParser(description="Voice Assistant pre-flight check")
    p.add_argument("--audio", action="store_true", help="Include 2-second mic recording test")
    args = p.parse_args()

    print("\nVoice Assistant -- Pre-flight diagnostics\n" + "=" * 44)
    check_python()
    check_packages()
    check_credentials()
    check_vosk_model()
    check_apis()
    check_audio()
    if args.audio:
        check_audio_recording()

    print("\n" + "=" * 44)
    hard = [r for r in _results if not r["passed"] and not r["warn"]]
    warn = [r for r in _results if not r["passed"] and     r["warn"]]
    if not hard:
        suffix = f"  ({len(warn)} optional items missing)" if warn else ""
        print(f"  {OK}  All required checks passed!{suffix}")
        print("\n  Ready to run:  python main.py")
    else:
        print(f"  {FAIL}  {len(hard)} check(s) failed -- fix before running.")
        for r in hard:
            print(f"    - {r['label']}")
        print("\n  See README.md for help.")
    print("=" * 44)


if __name__ == "__main__":
    main()
