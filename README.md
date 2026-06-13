# WiseVision — Boutique CCTV Analytics

Detect people in a shop video, keep stable IDs, label the **seller** vs
**clients**, count clients entering across a line, and detect
seller↔client interactions.

The work is provided as **two interchangeable methods** (same goal, different
tracker), plus helper tools.

## Repository layout

```
.
├── bot-sort-version/          # METHOD 1 — recommended
│   ├── bot-sort.py            #   the pipeline (BoT-SORT + OSNet ReID)
│   ├── requirements1.txt      #   dependencies
│   ├── reference_image/       #   reference photo(s) of the seller (used by ReID match)
│   └── solution1.txt          #   how this method works (diagrams)
│
├── bytetrack-version/         # METHOD 2 — baseline
│   ├── last.py                #   the pipeline (ByteTrack / yolox)
│   ├── requirements2.txt      #   dependencies
│   ├── ByteTrack/             #   the ByteTrack (yolox) tracker  [git submodule]
│   └── solution2.txt          #   how this method works (diagrams)
│
├── tools for help/            # standalone helper scripts
│   ├── pick_line.py           #   click two points to get counting-line coords
│   ├── concat_videos.py       #   join video clips
│   ├── test-detection.py      #   detection-only preview (no tracking)
│   └── video_info.py          #   print fps / size / frame count
│
├── prise_en_charge_client.ipynb   # separate notebook (standalone)
└── README.md
```

> **Models (`*.pt`) and videos (`*.mp4`) are git-ignored** — keep them locally.
> The only image committed is the seller reference in
> `bot-sort-version/reference_image/`.

## The two methods, briefly

| | **bot-sort-version** (Method 1) | **bytetrack-version** (Method 2) |
|---|---|---|
| Tracker | BoT-SORT + **OSNet ReID** (appearance + motion) | ByteTrack / yolox (**motion + IoU only**) |
| Seller ID | **persistence** (longest-present person) + optional ReID/photo + uniform | temporal lock (first person in a time window) |
| Counting | single line + **hysteresis band** (jitter/skip-robust) | single side-of-line test |
| Setup | `pip install` only | also needs the bundled **ByteTrack/yolox** |
| Status | **recommended** | baseline / comparison |

Full logic + diagrams: see `solution1.txt` and `solution2.txt`.

## Run it locally

Use one virtual environment per method (or a shared one).

### Method 1 — BoT-SORT
```bash
cd bot-sort-version
python -m venv venv && source venv/Scripts/activate      # Windows Git Bash
# install PyTorch (pick GPU or CPU — see comment in requirements1.txt), then:
pip install -r requirements1.txt
# place your video next to bot-sort.py (or edit SOURCE_VIDEO_PATH inside it)
python bot-sort.py            # writes shop_output_botsort.mp4
```
The OSNet ReID weights download automatically on first run.

### Method 2 — ByteTrack
```bash
cd bytetrack-version
git submodule update --init ByteTrack          # if not already populated
python -m venv venv && source venv/Scripts/activate
# install PyTorch (see requirements2.txt), then:
pip install -r requirements2.txt
pip install -r ByteTrack/requirements.txt      # yolox extras
# last.py expects ./ByteTrack relative to here, so run from THIS folder:
python last.py                # writes shop_output_last1.mp4
```

### Helper tools
```bash
cd "tools for help"
python pick_line.py           # get counting-line coordinates by clicking
python video_info.py          # inspect a video's fps/size/frames
python test-detection.py      # detection-only preview
```

## Notes
- Both pipelines target a **fixed boutique camera**; the counting line
  coordinates (`LINE1_*`) are tuned to that view — re-pick them with
  `pick_line.py` for a different camera.
- Appearance-based identity is limited by the camera's resolution/distance;
  see the closing note in `solution1.txt`.
