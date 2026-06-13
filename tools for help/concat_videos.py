import subprocess
import shutil
import os

VIDEOS = ["CRK01.mp4", "CRK02.mp4"]
OUTPUT = "CRK_full.mp4"
FILELIST = "_filelist_tmp.txt"

# Locate ffmpeg: PATH first, then imageio-ffmpeg bundle
ffmpeg = shutil.which("ffmpeg")
if ffmpeg is None:
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

if ffmpeg is None:
    raise RuntimeError(
        "ffmpeg not found. Install it:\n"
        "  Option 1: pip install imageio-ffmpeg\n"
        "  Option 2: download from https://ffmpeg.org and add to PATH"
    )

print(f"Using ffmpeg: {ffmpeg}")

for v in VIDEOS:
    if not os.path.exists(v):
        raise FileNotFoundError(f"Video not found: {v}")

with open(FILELIST, "w") as f:
    for v in VIDEOS:
        f.write(f"file '{v}'\n")

print(f"Concatenating {VIDEOS} → {OUTPUT} ...")
subprocess.run(
    [ffmpeg, "-f", "concat", "-safe", "0",
     "-i", FILELIST, "-c", "copy", OUTPUT],
    check=True
)

os.remove(FILELIST)

size_mb = os.path.getsize(OUTPUT) / (1024 ** 2)
print(f"Done. Output: {OUTPUT}  ({size_mb:.0f} MB)")
