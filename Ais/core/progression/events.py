# award() hook + level-up queue. Level boundary crossings are pushed onto a
# queue the toast UI drains; the feature colour is persisted onto the profile.
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import Ais.core.config as cfg
from . import orbs
from . import profile as _profile


Color = Tuple[float, float, float]


@dataclass
class LevelUp:
    name: str
    old_level: int
    new_level: int
    color: Color
    timestamp: float


_queue: Deque[LevelUp] = deque(maxlen=16)
_queue_lock = threading.Lock()

_last_award_t: Dict[str, float] = {}    # for rate-limited award sites
_RATE_LIMIT_S = 0.10

_global_last_award_t: float = 0.0       # monotonic seconds, any feature
_feature_last_seen_t: Dict[str, float] = {}   # monotonic seconds, per feature
_feature_last_gain: Dict[str, int] = {}        # last delta xp per feature, for hud float

_last_award_cursor: Optional[Tuple[float, float]] = None  # cursor pos of the last successful award
_CURSOR_EPS_PX = 0.5


def time_since_last_award() -> float:
    if _global_last_award_t == 0.0:
        return 1e9
    return time.monotonic() - _global_last_award_t


def time_since_feature(name: str) -> float:
    t = _feature_last_seen_t.get(_profile.canonical(name))
    if t is None:
        return 1e9
    return time.monotonic() - t


def recent_features(window_s: float = 12.0) -> List[Tuple[str, float]]:
    # [(name, seconds_ago)] for features touched within window_s, recent first
    now = time.monotonic()
    out: List[Tuple[str, float]] = []
    for name, t in _feature_last_seen_t.items():
        dt = now - t
        if dt <= window_s:
            out.append((name, dt))
    out.sort(key=lambda kv: kv[1])
    return out


def last_gain(name: str) -> int:
    return _feature_last_gain.get(_profile.canonical(name), 0)


def award(
    skill: str,
    xp: int,
    color: Optional[Color] = None,
    rate_limit: bool = False,
    cursor_pos: Optional[Tuple[float, float]] = None,
    orb_radius: float = 0.0,
) -> None:
    if xp <= 0 or not skill or _profile.is_placeholder_skill(skill):
        return

    # names containing '+' earn 10x XP
    if "+" in skill:
        xp *= 10

    # Skill identity is case-insensitive: bookkeeping and profile storage key off
    # the canonical name, while `skill` keeps the user's casing for display.
    key = _profile.canonical(skill)

    # skip if the cursor hasn't moved since the last award, but only for the
    # continuous brush (rate_limit); discrete clicks (box placement) always count
    global _last_award_cursor
    if rate_limit and cursor_pos is not None and _last_award_cursor is not None:
        if (abs(cursor_pos[0] - _last_award_cursor[0]) < _CURSOR_EPS_PX
                and abs(cursor_pos[1] - _last_award_cursor[1]) < _CURSOR_EPS_PX):
            return

    if rate_limit:
        now = time.monotonic()
        if now - _last_award_t.get(key, 0.0) < _RATE_LIMIT_S:
            return
        _last_award_t[key] = now

    if cursor_pos is not None:
        _last_award_cursor = (float(cursor_pos[0]), float(cursor_pos[1]))

    p = _profile.get_profile()
    p.register_name(skill)   # remember the display casing for this canonical skill
    before = p.skill_level(key)
    p.skills[key] = p.skills.get(key, 0) + xp
    after = p.skill_level(key)
    if color is not None:
        p.colors[key] = (float(color[0]), float(color[1]), float(color[2]))
    # earning XP un-hides a skill
    p.hidden.discard(key)

    if after > before:
        _push(LevelUp(
            name=skill,
            old_level=before,
            new_level=after,
            color=tuple(color) if color is not None else p.skill_color(key),
            timestamp=time.time(),
        ))

    # XP orbs fly from the cursor's brush/box rim into the HUD
    if cursor_pos is not None and cfg.settings.get("PERK_XP_ORBS", True):
        _oc = tuple(color) if color is not None else p.skill_color(key)
        _on = max(1, min(8, 1 + xp // 4))
        orbs.emit(float(cursor_pos[0]), float(cursor_pos[1]), _oc, _on, key, radius=orb_radius)

    now_mono = time.monotonic()
    global _global_last_award_t
    _global_last_award_t = now_mono
    _feature_last_seen_t[key] = now_mono
    _feature_last_gain[key] = xp
    _profile.mark_dirty()


# Passive XP: a slow trickle while a model trains, and a tiny bit each time a
# model paints a freshly scrolled-to slice. Paced by wall-clock, not callbacks.
TRAIN_INTERVAL_S = 2.5
TRAIN_XP = 1
INFER_INTERVAL_S = 0.35
INFER_XP = 1
_paced_last_t: Dict[str, float] = {}


def _paced(key: str, skill: str, xp: int, color: Optional[Color], interval: float,
           screen_xy: Optional[Tuple[float, float]], n_orbs: int, radius: float) -> None:
    if _profile.is_placeholder_skill(skill):
        return
    now = time.monotonic()
    if now - _paced_last_t.get(key, 0.0) < interval:
        return
    _paced_last_t[key] = now
    award(skill, xp, color)   # XP + HUD row; no cursor gate, no auto orbs
    if screen_xy is not None and cfg.settings.get("PERK_XP_ORBS", True):
        c = tuple(float(v) for v in color) if color is not None else _profile.get_profile().skill_color(skill)
        orbs.emit(float(screen_xy[0]), float(screen_xy[1]), c, n_orbs, _profile.canonical(skill), radius=radius)


def award_training(skill: str, color: Optional[Color] = None,
                   screen_xy: Optional[Tuple[float, float]] = None) -> None:
    if skill:
        _paced("train:" + _profile.canonical(skill), skill, TRAIN_XP, color, TRAIN_INTERVAL_S, screen_xy, 3, 8.0)


def award_inference(skill: str, color: Optional[Color] = None,
                    screen_xy: Optional[Tuple[float, float]] = None) -> None:
    # inference feeds an existing (annotation-made) skill; it never creates one
    if not skill or _profile.canonical(skill) not in _profile.get_profile().skills:
        return
    _paced("infer:" + _profile.canonical(skill), skill, INFER_XP, color, INFER_INTERVAL_S, screen_xy, 2, 6.0)


def _push(ev: LevelUp) -> None:
    with _queue_lock:
        _queue.append(ev)


def pop_level_up() -> Optional[LevelUp]:
    with _queue_lock:
        if _queue:
            return _queue.popleft()
    return None


def peek_pending() -> int:
    with _queue_lock:
        return len(_queue)
