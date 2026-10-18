"""Synthesise spoken versions of golden-set questions (English + Tamil) via the vertex-gateway.

Plain text only — Gemini TTS reads any style prefix aloud. One call at a time:
the gateway 429s on parallel TTS calls.
"""
import base64, json, time, wave, io
from pathlib import Path
import requests

GW = dict(l.strip().split("=", 1) for l in open(Path(__file__).with_name("gw.env")) if "=" in l)
MODEL, VOICE = "gemini-3.8-flash-tts", "Charon"
OUT = Path(__file__).with_name("audio"); OUT.mkdir(exist_ok=True)

# (id, theme, kind, english, tamil)
QUERIES = [
    ("fear1", "fear", "in", "What is fear?", "பயம் என்றால் என்ன?"),
    ("fear4", "fear", "in", "What is the relationship between thought and fear?", "எண்ணத்திற்கும் பயத்திற்கும் என்ன தொடர்பு?"),
    ("auth2", "authority", "in", "Can a guru actually help you understand yourself?", "ஒரு குரு உண்மையில் உங்களைப் புரிந்துகொள்ள உதவ முடியுமா?"),
    ("self1", "self-knowledge", "in", "What is self-knowledge?", "தன்னை அறிதல் என்றால் என்ன?"),
    ("self3", "self-knowledge", "in", "The observer is the observed. What does that mean?", "கவனிப்பவரே கவனிக்கப்படுவது. அதன் பொருள் என்ன?"),
    ("rel1", "relationship", "in", "What is the root of conflict in relationship?", "உறவில் ஏற்படும் முரண்பாட்டின் வேர் என்ன?"),
    ("aw2", "awareness", "in", "How is meditation different from concentration?", "தியானம் ஒருமுகப்படுத்தலிலிருந்து எப்படி வேறுபடுகிறது?"),
    ("th2", "thought", "in", "Why does the mind seek security through belief?", "மனம் ஏன் நம்பிக்கையின் மூலம் பாதுகாப்பைத் தேடுகிறது?"),
    ("love1", "love", "in", "What is love, and is it different from desire?", "அன்பு என்றால் என்ன, அது ஆசையிலிருந்து வேறுபட்டதா?"),
    ("cmp1", "comparison", "in", "Why do we compare ourselves to others?", "நாம் ஏன் நம்மை மற்றவர்களுடன் ஒப்பிடுகிறோம்?"),
    ("mem1", "memory", "in", "Why does the mind cling to the past?", "மனம் ஏன் கடந்த காலத்தைப் பற்றிக்கொள்கிறது?"),
    ("free1", "freedom", "in", "What is total freedom, and is it different from doing what one likes?", "முழுமையான சுதந்திரம் என்றால் என்ன, அது விரும்பியதைச் செய்வதிலிருந்து வேறுபட்டதா?"),
    ("off1", None, "off", "How do I install PostgreSQL on Ubuntu?", None),
    ("off2", None, "off", "Bitcoin price prediction for next year.", None),
    ("off3", None, "off", "Who won the 1998 football World Cup?", None),
    ("off4", None, "off", "Give me a recipe for banana bread.", None),
    ("off5", None, "off", "What is the capital of France?", None),
    ("off6", None, "off", "What is the best laptop for software development?", None),
]


def tts(text: str, path: Path) -> None:
    if path.exists():
        return
    body = {"contents": [{"role": "user", "parts": [{"text": text}]}],
            "generationConfig": {"responseModalities": ["AUDIO"],
                                 "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": VOICE}}}}}
    for attempt in range(12):
        r = requests.post(f"{GW['GEMINI_BASE']}/{MODEL}:generateContent", json=body, timeout=180,
                          headers={"x-goog-api-key": GW["GEMINI_API_KEY"]})
        if r.status_code == 200:
            b = base64.b64decode(r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]["data"])
            assert b[:4] == b"RIFF", "gateway did not return WAV"
            path.write_bytes(b)
            return
        wait = min(10 * (attempt + 1), 60)
        print(f"  {path.name}: HTTP {r.status_code}, retry in {wait}s")
        time.sleep(wait)
    raise SystemExit(f"TTS failed for {path.name}")


if __name__ == "__main__":
    for qid, theme, kind, en, ta in QUERIES:
        for lang, text in (("en", en), ("ta", ta)):
            if text:
                p = OUT / f"{qid}_{lang}.wav"
                tts(text, p)
                with wave.open(str(p)) as w:
                    print(f"{p.name}: {w.getnframes() / w.getframerate():.1f}s @ {w.getframerate()} Hz")
                time.sleep(2)
    json.dump([dict(id=q[0], theme=q[1], kind=q[2], en=q[3], ta=q[4]) for q in QUERIES],
              open(OUT / "queries.json", "w"), ensure_ascii=False, indent=1)
