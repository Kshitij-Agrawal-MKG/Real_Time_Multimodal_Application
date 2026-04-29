"""
generate_canned.py — Pre-generate Polly fallback audio clips.

Run once after setting up credentials.
Creates canned/ with raw PCM clips used when TTS fails or times out.

Usage:
    python generate_canned.py
    python generate_canned.py --voice Matthew --region eu-west-1
"""

import argparse
import sys
from pathlib import Path

_PHRASES = {
    "processing": "Just a moment, I am still processing.",
    "asr_error":  "Sorry, I did not catch that. Could you repeat?",
    "llm_error":  "Sorry, I am having trouble thinking right now.",
    "tts_error":  "There was an audio issue. Please try again.",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--voice",  default="Joanna",   help="Polly voice ID")
    p.add_argument("--region", default="us-east-1", help="AWS region")
    args = p.parse_args()

    try:
        import boto3
    except ImportError:
        print("ERROR: boto3 not installed.  Run: pip install boto3")
        sys.exit(1)

    try:
        polly = boto3.client("polly", region_name=args.region)
    except Exception as e:
        print(f"ERROR: Could not create Polly client: {e}")
        sys.exit(1)

    out = Path("canned")
    out.mkdir(exist_ok=True)
    print(f"Generating clips  voice={args.voice!r}  region={args.region!r}\n")

    for key, text in _PHRASES.items():
        try:
            r = polly.synthesize_speech(
                Text=text, VoiceId=args.voice, Engine="neural",
                OutputFormat="pcm", SampleRate="16000",
            )
            path = out / f"{key}.pcm"
            path.write_bytes(r["AudioStream"].read())
            print(f"  [OK]  canned/{key}.pcm  ({path.stat().st_size:,} bytes)")
        except Exception as e:
            print(f"  [FAIL]  {key}: {e}")

    print(f"\nDone. Clips saved to canned/")


if __name__ == "__main__":
    main()
