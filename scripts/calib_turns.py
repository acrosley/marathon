import sys
from transformers import AutoTokenizer
sys.path.insert(0, "/mnt/c/Users/acros/Projects/marathon/src")
from marathon.local_probe import _FILLER, _SYSTEM, _SEP, _PARITY_QUESTION
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B-FP8")
sep = tok.encode(_SEP.strip())[1:]
n = len(tok.encode(_SYSTEM, add_special_tokens=False))
per_user = len(sep) + len(tok.encode(f"user: Turn 10. {_FILLER} Reply 'ok'.", add_special_tokens=False))
per_asst = len(sep) + len(tok.encode("assistant: ok", add_special_tokens=False))
tail = len(sep) + len(tok.encode("\nassistant: <think>\n\n</think>\n\n", add_special_tokens=False))
print("sys", n, "user", per_user, "asst", per_asst, "tail", tail)
for target in (4096, 8192, 12288, 16384, 24576, 32768):
    T = round((target - n - tail - per_user) / (per_user + per_asst)) + 1
    print(target, "turns=", T, "approx_tokens=", n + tail + T*per_user + (T-1)*per_asst)
