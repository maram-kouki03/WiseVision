from dataclasses import dataclass
from typing import List, Optional, Dict

import time

import cv2
import numpy as np
np.float = float

import torch
from ultralytics import YOLO

import sys
sys.path.append('./ByteTrack')
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch

import supervision as sv

# =============================================================
#  CONFIG
# =============================================================
SOURCE_VIDEO_PATH          = "CRK_full.mp4"
TARGET_VIDEO_PATH          = "shop_output_last1.mp4"

MODEL_NAME                 = "yolo11m.pt"
CONF_SELLER                = 0.1
CONF_CLIENT                = 0.20
PERSON_CLASS_ID            = 0

SKIP                       = 3
SELLER_LOCK_WINDOW_SECONDS = 90
SELLER_RADIUS              = 300
ROLE_INHERIT_IOU_SELLER    = 0.1
ROLE_INHERIT_IOU_CLIENT    = 0.2
GHOST_TTL_FRAMES           = 300
CLIENT_RADIUS              = 150
SELLER_ALONE_SECONDS       = 5
INTERACTION_DIST           = 350

REQUIRE_GPU    = True
DEVICE         = "cuda:0" if torch.cuda.is_available() else "cpu"
HALF_PRECISION = torch.cuda.is_available()

LINE1_START = sv.Point(1705, 808)
LINE1_END   = sv.Point(1603, 1078)
LINE2_START = sv.Point(1696, 814)
LINE2_END   = sv.Point(1584, 1078)


# =============================================================
#  ByteTrack args
# =============================================================
@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh:        float = 0.1
    track_buffer:        int   = 300 
    match_thresh:        float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area:        float = 10.0
    mot20:               bool  = False


def detections2boxes(detections: sv.Detections) -> np.ndarray:
    return np.hstack((detections.xyxy, detections.confidence[:, np.newaxis]))


def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([t.tlbr for t in tracks], dtype=float)


def match_detections_with_tracks(detections: sv.Detections,
                                tracks: List[STrack]) -> List[Optional[int]]:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return [None] * len(detections)
    iou       = box_iou_batch(tracks2boxes(tracks), detections.xyxy)
    track2det = np.argmax(iou, axis=1)
    ids: List[Optional[int]] = [None] * len(detections)
    for ti, di in enumerate(track2det):
        if iou[ti, di] != 0:
            ids[di] = tracks[ti].track_id
    return ids


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
#  RoleTracker
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

        # Seller rescue: if Step 1 missed the seller but a CLIENT overlaps the
        # seller's last bbox by IoU > 0.85 for 3 consecutive frames, relabel it.
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
#  Helpers
# =============================================================
def is_in_side(point, ls: sv.Point, le: sv.Point) -> bool:
    cross = (le.x - ls.x) * (point[1] - ls.y) - (le.y - ls.y) * (point[0] - ls.x)
    return cross < 0


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

    byte_tracker = BYTETracker(BYTETrackerArgs())
    role_tracker = RoleTracker(
        iou_threshold_seller=ROLE_INHERIT_IOU_SELLER,
        iou_threshold_client=ROLE_INHERIT_IOU_CLIENT,
        ghost_ttl=GHOST_TTL_FRAMES,
        seller_lock_window_seconds=SELLER_LOCK_WINDOW_SECONDS,
        seller_radius=SELLER_RADIUS,
        client_conf_threshold=CONF_CLIENT,
    )

    video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
    fps        = video_info.fps
    generator  = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)

    prev_sides_l1:     Dict[int, bool]  = {}
    prev_sides_l2:     Dict[int, bool]  = {}
    pending_cross:     Dict[int, tuple] = {}
    entry_times:       Dict[int, float] = {}
    exit_times:        Dict[int, float] = {}
    last_enter_time:   float            = -9999.0
    pending_co_enters: Dict[int, float] = {}
    interaction_start: Dict[int, float] = {}
    seller_interactions: Dict[int, float] = {}  # every fid that interacted ≥ 5s
    in_count  = 0
    out_count = 0

    last_persons: List[RolePerson] = []
    processed_frame_idx = -1

    # Timing accumulators
    yolo_ms_total = 0.0   # pure YOLO inference (from results.speed)
    frame_ms_total = 0.0  # full per-processed-frame wall time
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

            # 2) ByteTrack
            tracks = byte_tracker.update(
                output_results=detections2boxes(detections),
                img_info=frame.shape,
                img_size=frame.shape,
            )
            detections.tracker_id = np.array(
                match_detections_with_tracks(detections, tracks), dtype=object)

            # 3) RoleTracker
            persons = role_tracker.update(
                det_xyxy=detections.xyxy,
                det_conf=detections.confidence,
                det_bytetrack_ids=detections.tracker_id,
                processed_frame_idx=processed_frame_idx,
                timestamp=timestamp,
            )

            # 4) Interaction detection — every fid close to seller for ≥ 5s counts,
            #    regardless of whether they have a recorded entry.
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

            # 5) Line crossing — dual-line confirmation
            DUAL_LINE_SECS  = 0.3
            new_inside_fids = []
            for p in persons:
                if p.role == "SELLER":
                    continue
                cx = (p.bbox[0] + p.bbox[2]) / 2
                cy = (p.bbox[1] + p.bbox[3]) / 2
                curr_l1 = is_in_side((cx, cy), LINE1_START, LINE1_END)
                curr_l2 = is_in_side((cx, cy), LINE2_START, LINE2_END)
                fid = p.first_pid

                if fid not in prev_sides_l1:
                    prev_sides_l1[fid] = curr_l1
                    prev_sides_l2[fid] = curr_l2
                    if not curr_l1 and not curr_l2:
                        new_inside_fids.append(fid)
                    continue

                old_l1 = prev_sides_l1[fid]; old_l2 = prev_sides_l2[fid]
                prev_sides_l1[fid] = curr_l1; prev_sides_l2[fid] = curr_l2

                line_events = []
                if old_l1 != curr_l1:
                    line_events.append(('L1', 'enter' if not curr_l1 else 'exit'))
                if old_l2 != curr_l2:
                    line_events.append(('L2', 'enter' if not curr_l2 else 'exit'))
                if not line_events:
                    continue

                for line_id, direction in line_events:
                    if fid in pending_cross:
                        pc_ts, pc_dir, pc_lines = pending_cross[fid]
                        expired = timestamp - pc_ts > DUAL_LINE_SECS
                        if expired or pc_dir != direction:
                            pending_cross[fid] = (timestamp, direction, {line_id})
                        elif line_id not in pc_lines:
                            pc_lines.add(line_id)
                            if len(pc_lines) == 2:
                                del pending_cross[fid]
                                if direction == 'enter':
                                    if fid not in exit_times:
                                        in_count += 1
                                        if fid not in entry_times:
                                            entry_times[fid] = pc_ts
                                        last_enter_time = timestamp
                                else:
                                    out_count += 1
                                    exit_times[fid] = timestamp
                    else:
                        pending_cross[fid] = (timestamp, direction, {line_id})

            for fid in new_inside_fids:
                if fid in entry_times or fid in exit_times:
                    continue
                if timestamp - last_enter_time <= DUAL_LINE_SECS:
                    pending_co_enters[fid] = timestamp

            active_fids = {p.first_pid for p in persons if p.role == "CLIENT"}
            for fid in list(pending_co_enters):
                ts = pending_co_enters.pop(fid)
                if fid in entry_times or fid in exit_times:
                    continue
                if fid in active_fids:
                    entry_times[fid] = ts
                    in_count += 1

            # 6) Draw + write
            draw_frame(frame, persons, in_count, len(seller_interactions))
            sink.write_frame(frame)
            last_persons = persons

            # Timing — sync GPU so the wall time reflects real inference
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            frame_ms_total += (time.perf_counter() - _frame_t0) * 1000.0
            yolo_ms_total  += sum(results[0].speed.values())  # pre+inf+post
            timed_frames   += 1

    del model, byte_tracker
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
