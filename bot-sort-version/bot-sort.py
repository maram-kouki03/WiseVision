#!/usr/bin/env python3
"""
CODE 2 - Full shop pipeline, but ByteTrack replaced by BoT-SORT (boxmot)
with a STRONG OSNet ReID model (osnet_x1_0_msmt17.pt).

The only changes vs your original file:
  - removed ByteTrack imports / helpers
  - the tracking step now runs boxmot BoT-SORT and maps its IDs back onto
    your detections (via the 'ind' column), then feeds them to RoleTracker
    exactly where ByteTrack IDs used to go.
RoleTracker, interaction detection, line crossing, drawing and timing are
unchanged.

Install:
    pip install boxmot
The strong OSNet weights download automatically on first run.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path

import os
import glob
import time

import cv2
import numpy as np
np.float = float

import torch
from ultralytics import YOLO

from onemetric.cv.utils.iou import box_iou_batch
import supervision as sv

# --- BoT-SORT from boxmot (use the version-robust factory) ---
from boxmot.trackers.tracker_zoo import create_tracker

# =============================================================
#  CONFIG
# =============================================================
SOURCE_VIDEO_PATH          = "CRK_full.mp4"
TARGET_VIDEO_PATH          = "shop_output_botsort.mp4"

MODEL_NAME                 = "yolo11m.pt"
CONF_SELLER                = 0.1
CONF_CLIENT                = 0.23
PERSON_CLASS_ID            = 0

# --- BoT-SORT / ReID ---
REID_MODEL                 = "osnet_ain_x1_0_msmt17.pt"   # OSNet-AIN: domain-generalizable, robust to camera/lighting gap
TRACK_BUFFER               = 180     # frames a lost track survives (tune to occlusion)
APPEARANCE_THRESH          = 0.3    # cosine-distance gate (lower = stricter); strict to avoid ID swaps
PROXIMITY_THRESH           = 0.8     # IoU gate before ReID; HIGH = let appearance rescue low-overlap matches
MATCH_THRESH               = 0.8     # IoU gate
CMC_METHOD                 = "ecc"   # camera-motion compensation: ecc | orb | sift | sof (cannot disable in this boxmot)
TRACK_HIGH_THRESH          = 0.4
TRACK_LOW_THRESH           = 0.1
NEW_TRACK_THRESH           = 0.2
APPEARANCE_EMA_ALPHA       = 0.95    # per-track feature momentum (lib default 0.9); higher = more stable, resists bad crops

# --- crop-quality gate: only feed clean crops to the ReID model (skip the rest) ---
# Low-quality crops (tiny, or wide "merged" boxes spanning two people) produce noisy
# OSNet embeddings that pollute track templates and cause ID swaps. Filtered detections
# simply get no BoT-SORT id this frame (tracker_id=None), exactly like an unmatched box.
CROP_MIN_HEIGHT            = 30      # px; crops shorter than this give unreliable embeddings
CROP_MIN_AREA              = 1200    # px^2; reject tiny boxes
CROP_MAX_ASPECT            = 1.3     # w/h; person boxes are tall, wide boxes are usually merged/partial

# --- Seller identification cue ---------------------------------------------
# Appearance (white uniform OR ReID photo-match) was tested and is UNRELIABLE on
# this footage: low-res distant warm-lit people give compressed embeddings, so
# clients score as "white" / as similar-to-seller as she does. The robust signal
# is PERSISTENCE: the seller works the whole session; clients come and go. We lock
# the SELLER role onto the identity that accumulates the most presence over time.
USE_SELLER_PERSISTENCE     = True
SELLER_MIN_FRAMES          = 40      # processed frames before an identity can be seller (~5s @ skip3/25fps); seller is on-screen from t=0
SELLER_LEAD_RATIO          = 1.5     # top identity must have >=1.5x the runner-up's presence to lock

# --- (experimental) ReID photo-match: compare each detection to a gallery of the
# seller's reference crops via the OSNet embedding. WEAK on this footage (clients
# reach ~0.74 sim), so TAKE is high and it's layered ABOVE persistence: it only
# overrides when very confident, else persistence/RoleTracker decide. Drop a
# clearer photo of the seller into SELLER_REF_DIR to (try to) improve it.
USE_SELLER_REID            = True
SELLER_REF_DIR             = "reference_image"  # folder of reference crop(s) of the seller
SELLER_REID_TAKE           = 0.80    # cosine-sim to claim/hold SELLER (high; clients reach ~0.74-0.77)
SELLER_REID_SWITCH         = 3       # consecutive frames a more-similar challenger must win to steal the role

# --- (legacy) white-uniform cue, kept for reference; disabled (too weak here) ---
USE_SELLER_UNIFORM         = False
SELLER_WHITE_TAKE          = 0.40    # torso white-ratio to claim/hold SELLER (seller ~0.5 on clean frames)
SELLER_WHITE_SWITCH        = 3       # consecutive frames a whiter challenger must win to steal the role
WHITE_S_MAX                = 80      # HSV saturation <= this (tuned for shop lighting / off-white)
WHITE_V_MIN                = 85      # HSV value (brightness) >= this; her white pull reads ~V90-110 (warm/dim light)

SKIP                       = 3
SELLER_LOCK_WINDOW_SECONDS = 90
SELLER_RADIUS              = 320
ROLE_INHERIT_IOU_SELLER    = 0.1
ROLE_INHERIT_IOU_CLIENT    = 0.2
GHOST_TTL_FRAMES           = 300
CLIENT_RADIUS              = 60
SELLER_ALONE_SECONDS       = 5
INTERACTION_DIST           = 350

REQUIRE_GPU    = True
DEVICE         = "cuda:0" if torch.cuda.is_available() else "cpu"
HALF_PRECISION = torch.cuda.is_available()

LINE1_START = sv.Point(1705, 808)
LINE1_END   = sv.Point(1603, 1078)
LINE2_START = sv.Point(1696, 814)
LINE2_END   = sv.Point(1584, 1078)

# Counting uses ONE line (LINE1) with a hysteresis band: a client is counted only
# when their position moves from >COUNT_MARGIN_PX on the outside to >COUNT_MARGIN_PX
# on the inside. The band rejects jitter near the line (no double counts) and the
# state machine needs no timing window (robust to frame-skips). Bigger = stricter.
COUNT_MARGIN_PX = 25.0


# =============================================================
#  Build the BoT-SORT tracker (robust to boxmot version kwargs)
# =============================================================
def _patch_appearance_ema(alpha: float):
    """Override BoT-SORT's hardcoded per-track feature-EMA momentum (default 0.9).
    Higher alpha = the appearance template adapts more slowly, so a single bad
    crop (occlusion / blur / neighbour bleed) barely perturbs a track's identity."""
    import boxmot.trackers.bbox.botsort.botsort_track as bt
    if getattr(bt.STrack, "_ema_patched", False):
        bt.STrack.alpha = alpha
        return
    _orig_init = bt.STrack.__init__
    def _init(self, *a, **k):
        _orig_init(self, *a, **k)
        self.alpha = alpha
    bt.STrack.__init__ = _init
    bt.STrack._ema_patched = True


def good_crop_mask(xyxy: np.ndarray) -> np.ndarray:
    """Boolean mask selecting detections whose crop is good enough for ReID:
    tall enough, large enough, and not an over-wide (likely merged) box."""
    if len(xyxy) == 0:
        return np.zeros((0,), dtype=bool)
    w = xyxy[:, 2] - xyxy[:, 0]
    h = xyxy[:, 3] - xyxy[:, 1]
    aspect = w / np.maximum(h, 1.0)
    return (h >= CROP_MIN_HEIGHT) & (w * h >= CROP_MIN_AREA) & (aspect <= CROP_MAX_ASPECT)


def torso_white_ratio(frame: np.ndarray, bbox) -> float:
    """Fraction of near-white pixels in the upper-torso region of a person box.
    The seller's white pull scores high (~0.7-0.8); colored clothing scores <0.1.
    Head and legs are skipped and a central column is used to avoid background."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return 0.0
    ty1 = max(0, y1 + int(0.15 * h)); ty2 = max(0, y1 + int(0.45 * h))   # upper shirt; avoid black pants
    tx1 = max(0, x1 + int(0.20 * w)); tx2 = max(0, x2 - int(0.20 * w))
    crop = frame[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = (hsv[:, :, 1] <= WHITE_S_MAX) & (hsv[:, :, 2] >= WHITE_V_MIN)
    return float(white.mean())


def build_seller_reid_template(reid_backend, ref_dir: str) -> Optional[np.ndarray]:
    """Embed every reference crop of the seller with the tracker's OSNet backend
    and average into one L2-normalized 'gallery' template. Returns None if the
    folder has no readable images (ReID matching then stays disabled)."""
    paths = sorted(glob.glob(os.path.join(ref_dir, "*.png")) +
                   glob.glob(os.path.join(ref_dir, "*.jpg")))
    embs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        f = np.asarray(reid_backend.get_features(np.array([[0, 0, w, h]], dtype=float), img)).reshape(-1)
        n = np.linalg.norm(f)
        if n > 0:
            embs.append(f / n)
    if not embs:
        return None
    t = np.mean(embs, axis=0)
    t /= (np.linalg.norm(t) + 1e-9)
    print(f"[SELLER-ReID] built template from {len(embs)} reference crop(s) in '{ref_dir}'")
    return t


def reid_seller_sims(persons: List["RolePerson"], frame: np.ndarray,
                     reid_backend, template: np.ndarray) -> List[float]:
    """Cosine similarity of each person's crop to the seller template (one ReID
    pass over all boxes). Higher = more like the reference photos of the seller."""
    if not persons:
        return []
    boxes = np.array([p.bbox for p in persons], dtype=float)
    feats = np.asarray(reid_backend.get_features(boxes, frame)).reshape(len(persons), -1)
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    return list(feats @ template)


class SellerUniformAnchor:
    """Keeps the SELLER label glued to the white-uniform person across id changes.

    Claims the whitest person above `take`; once a seller is held, only switches
    to a different person if that challenger stays distinctly whiter for
    `switch_patience` consecutive frames (so a client brushing a white surface,
    or a one-frame fluke, can't steal the role). Holds the current seller through
    brief uniform dips (turning around / partial occlusion)."""

    def __init__(self, take: float, switch_patience: int):
        self.take = take
        self.switch_patience = switch_patience
        self.seller_pid: Optional[int] = None
        self._chal_pid: Optional[int] = None
        self._chal_hits = 0

    def update(self, persons: List["RolePerson"], scores: List[float]) -> Optional["RolePerson"]:
        if not persons:
            return None
        i_top = int(np.argmax(scores))
        top_p, top_s = persons[i_top], scores[i_top]
        cur = next((p for p in persons if p.pid == self.seller_pid), None)

        # No live seller (first time, or her track was lost) -> claim whitest if clear.
        if cur is None:
            self._chal_pid, self._chal_hits = None, 0
            if top_s >= self.take:
                self.seller_pid = top_p.pid
                return top_p
            return None

        cur_s = scores[persons.index(cur)]
        # Keep the current seller while she is the whitest, nobody clears the bar,
        # or no challenger is *distinctly* whiter than her.
        if top_p.pid == cur.pid or top_s < self.take or top_s <= cur_s + 0.10:
            self._chal_hits = 0
            return cur
        # A different person is distinctly whiter -> require sustained evidence.
        if self._chal_pid == top_p.pid:
            self._chal_hits += 1
        else:
            self._chal_pid, self._chal_hits = top_p.pid, 1
        if self._chal_hits >= self.switch_patience:
            self.seller_pid = top_p.pid
            self._chal_hits = 0
            return top_p
        return cur


class SellerPersistenceAnchor:
    """Identify the seller as the most-persistently-present identity.

    The seller works the whole session; clients come and go. We accumulate
    per-identity presence (keyed by RoleTracker's stable ``first_pid``, which
    survives BoT-SORT id switches) and lock the SELLER role onto the identity
    that is clearly the longest-present, once it clears ``min_frames`` and leads
    the runner-up by ``lead_ratio``. The lock self-corrects: if another identity
    later overtakes by that margin, the role moves. The seller is only emitted
    when she is actually visible this frame."""

    def __init__(self, min_frames: int, lead_ratio: float):
        self.min_frames = min_frames
        self.lead_ratio = lead_ratio
        self.presence: Dict[int, int] = {}
        self.seller_fp: Optional[int] = None

    def update(self, persons: List["RolePerson"]) -> Optional["RolePerson"]:
        for p in persons:
            self.presence[p.first_pid] = self.presence.get(p.first_pid, 0) + 1
        if not persons:
            return None
        ranked = sorted(self.presence.items(), key=lambda kv: -kv[1])
        top_fp, top_n = ranked[0]
        second_n = ranked[1][1] if len(ranked) > 1 else 0
        if top_n >= self.min_frames and top_n >= self.lead_ratio * max(second_n, 1):
            if self.seller_fp != top_fp:
                print(f"[SELLER] locked identity first_pid={top_fp} "
                      f"(presence={top_n} vs runner-up {second_n})")
            self.seller_fp = top_fp
        if self.seller_fp is None:
            return None
        return next((p for p in persons if p.first_pid == self.seller_fp), None)


def build_botsort():
    _patch_appearance_ema(APPEARANCE_EMA_ALPHA)
    tuning = dict(
        track_high_thresh=TRACK_HIGH_THRESH,
        track_low_thresh=TRACK_LOW_THRESH,
        new_track_thresh=NEW_TRACK_THRESH,
        track_buffer=TRACK_BUFFER,
        match_thresh=MATCH_THRESH,
        proximity_thresh=PROXIMITY_THRESH,
        appearance_thresh=APPEARANCE_THRESH,
        cmc_method=CMC_METHOD,
    )
    # create_tracker builds the ReID backend from the weights path, moves it to
    # the requested device, applies half precision, and warms it up.
    return create_tracker(
        tracker_type="botsort",
        reid_weights=Path(REID_MODEL),
        device=torch.device(DEVICE),
        half=HALF_PRECISION,
        per_class=False,
        evolve_param_dict=tuning,
    )


# =============================================================
#  RolePerson
# =============================================================
@dataclass
class RolePerson:
    pid:             int
    bbox:            np.ndarray
    role:            str
    last_seen_frame: int
    bytetrack_id:    Optional[int] = None
    confidence:      float = 0.0
    first_pid:       int   = 0


# =============================================================
#  RoleTracker  (unchanged)
# =============================================================
class RoleTracker:
    def __init__(self, iou_threshold_seller: float, iou_threshold_client: float,
                ghost_ttl: int, seller_lock_window_seconds: float, seller_radius: float,
                client_conf_threshold: float = 0.2):
        self.iou_threshold_seller  = iou_threshold_seller
        self.iou_threshold_client  = iou_threshold_client
        self.ghost_ttl             = ghost_ttl
        self.seller_window         = seller_lock_window_seconds
        self.seller_radius         = seller_radius
        self.client_conf_threshold = client_conf_threshold
        self.persons: Dict[int, RolePerson] = {}
        self.next_pid      = 1
        self.seller_locked = False
        self.seller_pid: Optional[int] = None
        self._alone_since: Optional[float] = None
        self.evicted_cache:      List[dict] = []
        self.pid_chains:         Dict[int, List[int]] = {}
        self.bt_to_first_pid:    Dict[int, int] = {}
        self._rescue_count:      int = 0
        self._rescue_fp:         Optional[int] = None

    def _apply(self, p: RolePerson, bbox: np.ndarray, conf: float,
               bt_id, frame: int):
        p.bbox            = bbox.copy()
        p.confidence      = float(conf)
        p.last_seen_frame = frame
        p.bytetrack_id    = int(bt_id) if bt_id is not None else None

    def update(self, det_xyxy: np.ndarray, det_conf: np.ndarray,
               det_bytetrack_ids: np.ndarray,
               processed_frame_idx: int, timestamp: float) -> List[RolePerson]:
        n = len(det_xyxy)
        if n == 0:
            self._evict(processed_frame_idx)
            return []

        if n == 1 and self.seller_locked and self.seller_pid in self.persons:
            if self._alone_since is None:
                self._alone_since = timestamp
            elif timestamp - self._alone_since >= SELLER_ALONE_SECONDS:
                seller = self.persons[self.seller_pid]
                self._apply(seller, det_xyxy[0], det_conf[0],
                            det_bytetrack_ids[0], processed_frame_idx)
                self._evict(processed_frame_idx)
                return [seller]
        else:
            self._alone_since = None

        result:   List[Optional[RolePerson]] = [None] * n
        used_det: set = set()
        client_pids = [p for p in self.persons if p != self.seller_pid]

        if self.seller_locked and self.seller_pid in self.persons:
            seller = self.persons[self.seller_pid]
            scx = (seller.bbox[0] + seller.bbox[2]) / 2
            scy = (seller.bbox[1] + seller.bbox[3]) / 2
            iou_s  = box_iou_batch(det_xyxy, seller.bbox[np.newaxis])[:, 0]
            best_i = -1; best_iou = 0.0
            for i in range(n):
                if iou_s[i] <= self.iou_threshold_seller:
                    continue
                dcx = (det_xyxy[i][0] + det_xyxy[i][2]) / 2
                dcy = (det_xyxy[i][1] + det_xyxy[i][3]) / 2
                dist_to_seller = np.sqrt((dcx - scx) ** 2 + (dcy - scy) ** 2)
                if dist_to_seller > self.seller_radius:
                    continue
                closer_client = False
                for cpid in client_pids:
                    cp  = self.persons[cpid]
                    ccx = (cp.bbox[0] + cp.bbox[2]) / 2
                    ccy = (cp.bbox[1] + cp.bbox[3]) / 2
                    if np.sqrt((dcx - ccx) ** 2 + (dcy - ccy) ** 2) < dist_to_seller:
                        closer_client = True; break
                if closer_client:
                    continue
                if iou_s[i] > best_iou:
                    best_iou = iou_s[i]; best_i = i
            if best_i >= 0:
                self._apply(seller, det_xyxy[best_i], det_conf[best_i],
                            det_bytetrack_ids[best_i], processed_frame_idx)
                result[best_i] = seller; used_det.add(best_i)
            else:
                best_dist = float(self.seller_radius); best_i = -1
                for i in range(n):
                    dcx  = (det_xyxy[i][0] + det_xyxy[i][2]) / 2
                    dcy  = (det_xyxy[i][1] + det_xyxy[i][3]) / 2
                    dist = np.sqrt((dcx - scx) ** 2 + (dcy - scy) ** 2)
                    if dist >= best_dist:
                        continue
                    near_client = False
                    for cpid in client_pids:
                        cp  = self.persons[cpid]
                        ccx = (cp.bbox[0] + cp.bbox[2]) / 2
                        ccy = (cp.bbox[1] + cp.bbox[3]) / 2
                        if np.sqrt((dcx - ccx) ** 2 + (dcy - ccy) ** 2) < CLIENT_RADIUS:
                            near_client = True; break
                    if near_client:
                        continue
                    best_dist = dist; best_i = i
                if best_i >= 0:
                    self._apply(seller, det_xyxy[best_i], det_conf[best_i],
                                det_bytetrack_ids[best_i], processed_frame_idx)
                    result[best_i] = seller; used_det.add(best_i)

        free_dets = [i for i in range(n)
                     if i not in used_det and det_conf[i] >= self.client_conf_threshold]
        if client_pids and free_dets:
            free_xyxy   = det_xyxy[free_dets]
            cand_bboxes = np.stack([self.persons[p].bbox for p in client_pids])
            iou_mat     = box_iou_batch(free_xyxy, cand_bboxes)
            assigned    = [False] * len(client_pids)
            order       = np.argsort(-iou_mat.max(axis=1))
            for fi in order:
                i = free_dets[fi]
                dcx = (det_xyxy[i][0] + det_xyxy[i][2]) / 2
                dcy = (det_xyxy[i][1] + det_xyxy[i][3]) / 2
                for j in np.argsort(-iou_mat[fi]):
                    if iou_mat[fi, j] <= self.iou_threshold_client:
                        break
                    if assigned[j]:
                        continue
                    p   = self.persons[client_pids[j]]
                    pcx = (p.bbox[0] + p.bbox[2]) / 2
                    pcy = (p.bbox[1] + p.bbox[3]) / 2
                    if np.sqrt((dcx - pcx) ** 2 + (dcy - pcy) ** 2) > CLIENT_RADIUS:
                        continue
                    self._apply(p, det_xyxy[i], det_conf[i],
                                det_bytetrack_ids[i], processed_frame_idx)
                    if p.bytetrack_id is not None:
                        self.bt_to_first_pid[p.bytetrack_id] = p.first_pid
                    result[i] = p; assigned[j] = True; used_det.add(i); break

            free_dets = [i for i in range(n) if i not in used_det]
            for i in free_dets:
                dcx = (det_xyxy[i][0] + det_xyxy[i][2]) / 2
                dcy = (det_xyxy[i][1] + det_xyxy[i][3]) / 2
                best_dist = float(CLIENT_RADIUS); best_j = -1
                for j, pid in enumerate(client_pids):
                    if assigned[j]:
                        continue
                    p   = self.persons[pid]
                    pcx = (p.bbox[0] + p.bbox[2]) / 2
                    pcy = (p.bbox[1] + p.bbox[3]) / 2
                    dist = np.sqrt((dcx - pcx) ** 2 + (dcy - pcy) ** 2)
                    if dist < best_dist:
                        best_dist = dist; best_j = j
                if best_j >= 0:
                    p = self.persons[client_pids[best_j]]
                    self._apply(p, det_xyxy[i], det_conf[i],
                                det_bytetrack_ids[i], processed_frame_idx)
                    if p.bytetrack_id is not None:
                        self.bt_to_first_pid[p.bytetrack_id] = p.first_pid
                    result[i] = p; assigned[best_j] = True; used_det.add(i)

        for i in range(n):
            if result[i] is not None:
                continue
            is_seller_candidate = (not self.seller_locked
                                   and timestamp <= self.seller_window)
            if not is_seller_candidate and det_conf[i] < self.client_conf_threshold:
                continue
            dcx_pre = (det_xyxy[i][0] + det_xyxy[i][2]) / 2
            dcy_pre = (det_xyxy[i][1] + det_xyxy[i][3]) / 2
            duplicate = False
            for j in range(n):
                if result[j] is None:
                    continue
                jcx  = (det_xyxy[j][0] + det_xyxy[j][2]) / 2
                jcy  = (det_xyxy[j][1] + det_xyxy[j][3]) / 2
                dist = np.sqrt((dcx_pre - jcx) ** 2 + (dcy_pre - jcy) ** 2)
                iou  = box_iou_batch(det_xyxy[i][np.newaxis],
                                     det_xyxy[j][np.newaxis])[0, 0]
                if dist < CLIENT_RADIUS or iou > 0.7:
                    duplicate = True; break
            if duplicate:
                continue
            new_pid = self.next_pid; self.next_pid += 1
            if not self.seller_locked and timestamp <= self.seller_window:
                role = "SELLER"; self.seller_locked = True
                self.seller_pid = new_pid; first_pid = new_pid
            else:
                role = "CLIENT"
                dcx = dcx_pre; dcy = dcy_pre
                first_pid = new_pid
                bt_id = (int(det_bytetrack_ids[i])
                         if det_bytetrack_ids[i] is not None else None)
                matched_first_pids = {p.first_pid for p in result if p is not None}

                if first_pid == new_pid and bt_id is not None:
                    if bt_id in self.bt_to_first_pid:
                        candidate = self.bt_to_first_pid[bt_id]
                        if candidate not in matched_first_pids:
                            first_pid = candidate

                if first_pid == new_pid:
                    best_ev_dist = float(CLIENT_RADIUS); best_ev_k = -1
                    for k, ev in enumerate(self.evicted_cache):
                        if ev['first_pid'] in matched_first_pids:
                            continue
                        ecx  = (ev['bbox'][0] + ev['bbox'][2]) / 2
                        ecy  = (ev['bbox'][1] + ev['bbox'][3]) / 2
                        dist = np.sqrt((dcx - ecx) ** 2 + (dcy - ecy) ** 2)
                        if dist < best_ev_dist:
                            best_ev_dist = dist; best_ev_k = k
                    if best_ev_k >= 0:
                        first_pid = self.evicted_cache[best_ev_k]['first_pid']
                        self.evicted_cache.pop(best_ev_k)

                if first_pid == new_pid:
                    best_stale_dist = CLIENT_RADIUS * 2.0; best_stale_pid = None
                    for cpid in client_pids:
                        if cpid not in self.persons:
                            continue
                        pa = self.persons[cpid]
                        if pa.last_seen_frame == processed_frame_idx:
                            continue
                        if pa.first_pid in matched_first_pids:
                            continue
                        ecx  = (pa.bbox[0] + pa.bbox[2]) / 2
                        ecy  = (pa.bbox[1] + pa.bbox[3]) / 2
                        dist = np.sqrt((dcx - ecx) ** 2 + (dcy - ecy) ** 2)
                        if dist < best_stale_dist:
                            best_stale_dist = dist; best_stale_pid = cpid
                    if best_stale_pid is not None:
                        first_pid = self.persons[best_stale_pid].first_pid
                        del self.persons[best_stale_pid]

                if first_pid == new_pid:
                    det_box = det_xyxy[i][np.newaxis]
                    best_score = 0.0; best_fp = None; best_ev_k = None; best_ghost = None
                    for k, ev in enumerate(self.evicted_cache):
                        if ev['first_pid'] in matched_first_pids:
                            continue
                        iou_val = box_iou_batch(det_box, ev['bbox'][np.newaxis])[0, 0]
                        ecx  = (ev['bbox'][0] + ev['bbox'][2]) / 2
                        ecy  = (ev['bbox'][1] + ev['bbox'][3]) / 2
                        dist = np.sqrt((dcx - ecx) ** 2 + (dcy - ecy) ** 2)
                        score = iou_val if iou_val > 0.85 else (1.0 if dist < 50 else 0.0)
                        if score > best_score:
                            best_score = score; best_fp = ev['first_pid']
                            best_ev_k = k; best_ghost = None
                    for cpid in client_pids:
                        if cpid not in self.persons:
                            continue
                        pa = self.persons[cpid]
                        if pa.last_seen_frame == processed_frame_idx:
                            continue
                        if pa.first_pid in matched_first_pids:
                            continue
                        iou_val = box_iou_batch(det_box, pa.bbox[np.newaxis])[0, 0]
                        ecx  = (pa.bbox[0] + pa.bbox[2]) / 2
                        ecy  = (pa.bbox[1] + pa.bbox[3]) / 2
                        dist = np.sqrt((dcx - ecx) ** 2 + (dcy - ecy) ** 2)
                        score = iou_val if iou_val > 0.85 else (1.0 if dist < 50 else 0.0)
                        if score > best_score:
                            best_score = score; best_fp = pa.first_pid
                            best_ev_k = None; best_ghost = cpid
                    if best_fp is not None:
                        first_pid = best_fp
                        if best_ev_k is not None:
                            self.evicted_cache.pop(best_ev_k)
                        elif best_ghost is not None:
                            del self.persons[best_ghost]

                if first_pid not in self.pid_chains:
                    self.pid_chains[first_pid] = [first_pid]
                if new_pid not in self.pid_chains[first_pid]:
                    self.pid_chains[first_pid].append(new_pid)

            new_bt = (int(det_bytetrack_ids[i])
                      if det_bytetrack_ids[i] is not None else None)
            new_p = RolePerson(
                pid=new_pid, bbox=det_xyxy[i].copy(), role=role,
                last_seen_frame=processed_frame_idx, bytetrack_id=new_bt,
                confidence=float(det_conf[i]), first_pid=first_pid,
            )
            if role == "CLIENT" and new_bt is not None:
                self.bt_to_first_pid[new_bt] = first_pid
            self.persons[new_pid] = new_p
            result[i] = new_p

        self._evict(processed_frame_idx)
        out = [p for p in result if p is not None]

        if self.seller_locked and self.seller_pid in self.persons:
            seller = self.persons[self.seller_pid]
            if not any(p.pid == self.seller_pid for p in out):
                rescue_found = False
                for idx, p in enumerate(out):
                    if p.role != 'CLIENT':
                        continue
                    iou = box_iou_batch(p.bbox[np.newaxis], seller.bbox[np.newaxis])[0, 0]
                    if iou > 0.85:
                        rescue_found = True
                        if self._rescue_fp == p.first_pid:
                            self._rescue_count += 1
                        else:
                            self._rescue_fp = p.first_pid; self._rescue_count = 1
                        if self._rescue_count >= 3:
                            self._apply(seller, p.bbox, p.confidence,
                                        p.bytetrack_id, processed_frame_idx)
                            if p.pid in self.persons:
                                del self.persons[p.pid]
                            out[idx] = seller
                            self._rescue_count = 0; self._rescue_fp = None
                        break
                if not rescue_found:
                    self._rescue_count = 0; self._rescue_fp = None
            else:
                self._rescue_count = 0; self._rescue_fp = None
        return out

    def _evict(self, current_frame: int):
        to_drop = [pid for pid, p in self.persons.items()
                   if pid != self.seller_pid
                   and current_frame - p.last_seen_frame > self.ghost_ttl]
        for pid in to_drop:
            p = self.persons[pid]
            self.evicted_cache.append({
                'first_pid': p.first_pid, 'bbox': p.bbox.copy(), 'frame': current_frame,
            })
            del self.persons[pid]
        self.evicted_cache = [e for e in self.evicted_cache
                              if current_frame - e['frame'] <= 300]


# =============================================================
#  Helpers  (unchanged)
# =============================================================
def is_in_side(point, ls: sv.Point, le: sv.Point) -> bool:
    cross = (le.x - ls.x) * (point[1] - ls.y) - (le.y - ls.y) * (point[0] - ls.x)
    return cross < 0


def signed_dist_to_line(point, ls: sv.Point, le: sv.Point) -> float:
    """Signed perpendicular distance (px) from point to the line. Sign matches
    is_in_side (cross<0 => negative side). 'Inside the shop' is the positive side
    (the original 'enter' direction), so d>0 means inside."""
    dx, dy = le.x - ls.x, le.y - ls.y
    cross = dx * (point[1] - ls.y) - dy * (point[0] - ls.x)
    return cross / (np.hypot(dx, dy) + 1e-9)


def draw_frame(frame: np.ndarray, persons: List[RolePerson],
               in_count: int, interaction_count: int) -> np.ndarray:
    SELLER_COLOR = (0, 0, 255)
    CLIENT_COLOR = (0, 200, 0)
    for p in persons:
        x1, y1, x2, y2 = map(int, p.bbox)
        color = SELLER_COLOR if p.role == "SELLER" else CLIENT_COLOR
        label = "SELLER" if p.role == "SELLER" else f"CLIENT #{p.first_pid}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.line(frame, (LINE1_START.x, LINE1_START.y), (LINE1_END.x, LINE1_END.y),
             (0, 200, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (LINE2_START.x, LINE2_START.y), (LINE2_END.x, LINE2_END.y),
             (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Clients: {in_count}  Interactions: {interaction_count}",
                (LINE1_START.x - 200, LINE1_START.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return frame


# =============================================================
#  Main
# =============================================================
def main():
    print("=" * 50)
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)} | CUDA {torch.version.cuda}")
    else:
        if REQUIRE_GPU:
            raise RuntimeError("[GPU] NO CUDA.")
        print("[GPU] running on CPU.")
    print("=" * 50)

    model = YOLO(MODEL_NAME)
    model.to(DEVICE)
    model.fuse()

    # ---- BoT-SORT with strong OSNet (replaces ByteTrack) ----
    botsort = build_botsort()

    role_tracker = RoleTracker(
        iou_threshold_seller=ROLE_INHERIT_IOU_SELLER,
        iou_threshold_client=ROLE_INHERIT_IOU_CLIENT,
        ghost_ttl=GHOST_TTL_FRAMES,
        seller_lock_window_seconds=SELLER_LOCK_WINDOW_SECONDS,
        seller_radius=SELLER_RADIUS,
        client_conf_threshold=CONF_CLIENT,
    )

    seller_anchor  = SellerUniformAnchor(SELLER_WHITE_TAKE, SELLER_WHITE_SWITCH)
    seller_persist = SellerPersistenceAnchor(SELLER_MIN_FRAMES, SELLER_LEAD_RATIO)

    # ReID seller template (built from reference crops via the tracker's OSNet backend)
    seller_template   = build_seller_reid_template(botsort.model, SELLER_REF_DIR) if USE_SELLER_REID else None
    seller_reid_anchor = SellerUniformAnchor(SELLER_REID_TAKE, SELLER_REID_SWITCH)  # generic score anchor (sim scores)
    if USE_SELLER_REID and seller_template is None:
        print(f"[SELLER-ReID] no reference crops in '{SELLER_REF_DIR}' -> ReID matching disabled")

    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    fps        = video_info.fps
    generator  = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    side_state:        Dict[int, str]   = {}   # fid -> 'in' | 'out' (last resolved side, with hysteresis)
    entry_times:       Dict[int, float] = {}
    exit_times:        Dict[int, float] = {}
    interaction_start: Dict[int, float] = {}
    seller_interactions: Dict[int, float] = {}
    in_count  = 0
    out_count = 0

    last_persons: List[RolePerson] = []
    processed_frame_idx = -1

    yolo_ms_total = 0.0
    frame_ms_total = 0.0
    timed_frames = 0

    with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
        for frame_idx, frame in enumerate(generator):

            if frame_idx % SKIP != 0:
                draw_frame(frame, last_persons, in_count,
                           len(seller_interactions))
                sink.write_frame(frame)
                continue

            processed_frame_idx += 1
            timestamp = frame_idx / fps
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _frame_t0 = time.perf_counter()

            # 1) YOLO
            results    = model(frame, conf=CONF_SELLER, classes=[PERSON_CLASS_ID],
                               iou=0.5, imgsz=640, device=DEVICE,
                               half=HALF_PRECISION, verbose=False)
            detections = sv.Detections(
                xyxy=results[0].boxes.xyxy.cpu().numpy(),
                confidence=results[0].boxes.conf.cpu().numpy(),
                class_id=results[0].boxes.cls.cpu().numpy().astype(int),
            )

            # 2) BoT-SORT (strong OSNet) -> per-detection track ids
            #    Only clean crops are fed to the tracker so the ReID embeddings
            #    stay reliable; keep_idx maps filtered rows back to detections.
            n_det    = len(detections.xyxy)
            keep_idx = np.nonzero(good_crop_mask(detections.xyxy))[0] if n_det else np.empty((0,), int)
            if len(keep_idx) > 0:
                dets = np.hstack([
                    detections.xyxy[keep_idx],
                    detections.confidence[keep_idx, None],
                    detections.class_id[keep_idx, None].astype(float),
                ])  # K x 6 (x1,y1,x2,y2,conf,cls)
            else:
                dets = np.empty((0, 6))

            tracks = botsort.update(dets, frame)  # M x 8 (...,id,conf,cls,ind)

            # map BoT-SORT ids back onto the ORIGINAL detections via keep_idx[ind]
            tracker_ids = [None] * n_det
            if tracks is not None and len(tracks) > 0:
                for tr in tracks:
                    j = int(tr[7])                    # index into the filtered dets
                    if 0 <= j < len(keep_idx):
                        tracker_ids[int(keep_idx[j])] = int(tr[4])
            detections.tracker_id = np.array(tracker_ids, dtype=object)

            # 3) RoleTracker (unchanged) - now fed BoT-SORT ids
            persons = role_tracker.update(
                det_xyxy=detections.xyxy,
                det_conf=detections.confidence,
                det_bytetrack_ids=detections.tracker_id,
                processed_frame_idx=processed_frame_idx,
                timestamp=timestamp,
            )

            # 3b) Seller anchors: a SUPPLEMENT to RoleTracker, not a replacement.
            #     RoleTracker's own seller logic always runs and stays in effect.
            #     Cues are layered by reliability and each only steps in when
            #     CONFIDENT: ReID photo-match -> persistence -> white uniform.
            #     The first confident cue wins; otherwise RoleTracker's roles stand.
            # Evaluate every enabled cue (persistence MUST run each frame to keep
            # its presence counts current as a fallback), then pick by priority.
            reid_seller = persist_seller = uniform_seller = None
            if persons and USE_SELLER_REID and seller_template is not None:
                sims = reid_seller_sims(persons, frame, botsort.model, seller_template)
                reid_seller = seller_reid_anchor.update(persons, sims)
            if persons and USE_SELLER_PERSISTENCE:
                persist_seller = seller_persist.update(persons)
            if persons and USE_SELLER_UNIFORM:
                uniform_seller = seller_anchor.update(persons, [torso_white_ratio(frame, p.bbox) for p in persons])

            seller_p = None; seller_key = None; key = None
            if reid_seller is not None:
                seller_p, key, seller_key = reid_seller, (lambda p: p.pid), reid_seller.pid
            elif persist_seller is not None:
                seller_p, key, seller_key = persist_seller, (lambda p: p.first_pid), persist_seller.first_pid
            elif uniform_seller is not None:
                seller_p, key, seller_key = uniform_seller, (lambda p: p.pid), uniform_seller.pid

            if seller_p is not None:
                # Confident seller -> correct the roles and align RoleTracker's pointer.
                for p in persons:
                    p.role = "SELLER" if key(p) == seller_key else "CLIENT"
                role_tracker.seller_pid    = seller_p.pid
                role_tracker.seller_locked = True
            # else: no confident anchor -> RoleTracker's own roles stand (unchanged).

            # 4) Interaction detection (unchanged)
            INTERACTION_MIN_SECS = 5.0
            seller_person   = next((p for p in persons if p.role == "SELLER"), None)
            currently_close = set()
            if seller_person is not None:
                sx1, sy1, sx2, sy2 = seller_person.bbox
                for p in persons:
                    if p.role != "CLIENT":
                        continue
                    fid = p.first_pid
                    if fid in seller_interactions:
                        continue
                    cx1, cy1, cx2, cy2 = p.bbox
                    gap_x = max(0, max(cx1, sx1) - min(cx2, sx2))
                    gap_y = max(0, max(cy1, sy1) - min(cy2, sy2))
                    if np.sqrt(gap_x ** 2 + gap_y ** 2) < INTERACTION_DIST:
                        currently_close.add(fid)
                        if fid not in interaction_start:
                            interaction_start[fid] = timestamp
                        elif timestamp - interaction_start[fid] >= INTERACTION_MIN_SECS:
                            seller_interactions[fid] = interaction_start.pop(fid)
            for fid in list(interaction_start):
                if fid not in currently_close:
                    del interaction_start[fid]

            # 5) Line crossing: single-line hysteresis state machine.
            #    Count a client once when its position moves from clearly-outside
            #    (d < -MARGIN) to clearly-inside (d > +MARGIN) of LINE1. The margin
            #    band rejects jitter near the line; comparing last-vs-current side
            #    needs no timing window, so frame-skips can't break a crossing.
            for p in persons:
                if p.role == "SELLER":
                    continue
                fid = p.first_pid
                cx = (p.bbox[0] + p.bbox[2]) / 2
                cy = (p.bbox[1] + p.bbox[3]) / 2
                d  = signed_dist_to_line((cx, cy), LINE1_START, LINE1_END)

                if d > COUNT_MARGIN_PX:
                    side = "in"
                elif d < -COUNT_MARGIN_PX:
                    side = "out"
                else:
                    side = side_state.get(fid)        # inside the band: hold last side

                prev = side_state.get(fid)
                if side is not None:
                    side_state[fid] = side
                if prev is None or side is None or side == prev:
                    continue                          # first sighting or no real transition

                if prev == "out" and side == "in":
                    in_count += 1
                    entry_times[fid] = timestamp
                elif prev == "in" and side == "out":
                    out_count += 1
                    exit_times[fid] = timestamp

            # 6) Draw + write
            draw_frame(frame, persons, in_count, len(seller_interactions))
            sink.write_frame(frame)
            last_persons = persons

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            frame_ms_total += (time.perf_counter() - _frame_t0) * 1000.0
            yolo_ms_total  += sum(results[0].speed.values())
            timed_frames   += 1

    del model, botsort
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n========== RESULT ==========")
    print(f"Clients entered  : {in_count}")
    print(f"Interactions     : {len(seller_interactions)}")
    print(f"Output video     : {TARGET_VIDEO_PATH}")
    if timed_frames > 0:
        print("---------- TIMING ----------")
        print(f"Processed frames : {timed_frames}")
        print(f"YOLO inference   : {yolo_ms_total / timed_frames:.1f} ms/frame "
              f"(pre+inf+post)")
        print(f"Full pipeline    : {frame_ms_total / timed_frames:.1f} ms/frame "
              f"({timed_frames / (frame_ms_total / 1000.0):.1f} FPS)")
    print("============================")


if __name__ == "__main__":
    main()