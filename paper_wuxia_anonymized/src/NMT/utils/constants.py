from pathlib import Path

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1
