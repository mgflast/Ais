# Living background: soft feature-colour shapes SPAWNED by your annotations.
#
# There is no pre-built pool and no XP/recency tracking. Every annotation action
# (a brush tick, a box, a copy) drops ONE new shape, which is given:
#   - a colour: the active tool's colour, with a little hue jitter. Rarely a
#     sibling's colour (another feature in the tomogram) or the inverted colour.
#   - a random size and velocity
#   - a fade-in: it starts invisible and eases in (it doesn't pop), drifts, and
#     eventually fades back out and dies.
# So the field is just a short-lived echo of what you are drawing. It is empty
# until you act. Opt-in via the Background cosmetic.
#
# Styles (from the equipped cosmetic's params):
#   "blob"   - big soft gaussian washes (Aurora)
#   "bokeh"  - small crisp discs that gently avoid the cursor
#
# Rendered in a GL pre-pass BEHIND the tomogram (fills the papery margins), mostly
# screen-space with a slight camera parallax. The GL draw lives in
# Renderer.render_background; this module hands it, each frame, a list of
# (x, y, radius, colour, angle, alpha) + base colour + intensity + a shape id.
from __future__ import annotations

import colorsys
import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import Ais.core.config as cfg
from . import cosmetics


Color = Tuple[float, float, float]

PARALLAX = 0.075        # subtle scene parallax 
_SHAPE = {"blob": 0, "bokeh": 2}

# --- spawn behaviour (tunable) ---------------------------------------------
_SPAWN_MIN_DT = 0.5      # min seconds between brush spawns; discrete actions (box/copy) bypass it
_MAX_BLOBS = 200         # safety cap on live shapes
_OTHER_CHANCE = 0.05     # spawn in a sibling feature's colour instead of the active one
_INVERT_CHANCE = 0.03    # spawn in the inverted active colour
_HUE_JITTER = 0.04       # per-shape hue wander around its colour
_LIFE = (4.0, 9.0)       # base life in seconds (multiplied by the cosmetic's life_mul)
_FADE_IN = 0.5           # ease-in time (s) - the shape appears gradually
_FADE_OUT_FRAC = 0.35    # the last this-fraction of life eases back out
_FADE_OUT_SHRINK = 1.0 / 3.0   # radius shrinks to this fraction across the fade-out
_TAU_RECOLOR = 0.67      # ~2s transition when a level-up recolours the field

# --- bokeh cursor avoidance (a gentle, heavily damped push - calm) ---------
_AVOID_R = 200.0
_AVOID_R2 = _AVOID_R * _AVOID_R
_AVOID_ACCEL = 160.0
_AVOID_DAMP = 0.02           # per-second velocity retention (heavy)
_AVOID_BRUSH_RANGE = 2.0     # while the LMB is held (brushing): range x5
_AVOID_BRUSH_STRENGTH = 2.0  # ...and strength x1.5
_DRIFT = (4.0, 11.0)         # bokeh baseline drift speed (px/s); also the per-blob
                             # damping threshold - only speed ABOVE it is damped away


@dataclass
class _Blob:
    x: float
    y: float
    vx: float
    vy: float
    r: float
    color: Color
    color_target: Color
    angle: float
    spin: float
    breathe: float
    phase: float        # random [0, 2pi) - aurora breathe offset
    age: float
    life: float
    drift: float        # bokeh baseline drift speed / damping threshold (px/s)


_blobs: List[_Blob] = []
_cur_style = ""
_screen_w = 1
_screen_h = 1
_last_spawn_t = 0.0
_bt = 0.0                 # time accumulator for the aurora "breathe"


def _soft(v: float) -> float:
    return v + (1.0 - v) * 0.12


def _soft_color(c) -> Color:
    return (_soft(float(c[0])), _soft(float(c[1])), _soft(float(c[2])))


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _hue_jitter(c: Color, amp: float = _HUE_JITTER) -> Color:
    h, s, v = colorsys.rgb_to_hsv(*c)
    h = (h + random.uniform(-amp, amp)) % 1.0
    s = _clamp(s + random.uniform(-0.06, 0.06), 0.0, 1.0)
    v = _clamp(v + random.uniform(-0.05, 0.05), 0.0, 1.0)
    return colorsys.hsv_to_rgb(h, s, v)


def _params() -> dict:
    return cosmetics.params(cosmetics.BACKGROUND)


def _siblings(active: Color) -> List[Color]:
    # The OTHER features in the current tomogram (annotation context). We never
    # reach into the model tab here - spawns come from annotation actions.
    frame_ = getattr(cfg, "se_active_frame", None)
    if frame_ is None:
        return []
    out: List[Color] = []
    for f in getattr(frame_, "features", []) or []:
        c = getattr(f, "colour", None)
        if c is None:
            continue
        c = (float(c[0]), float(c[1]), float(c[2]))
        if c != active:
            out.append(c)
    return out


def spawn(active_colour, throttle: bool = True, count: int = 1, siblings=None) -> bool:
    """A user action drops `count` new shapes. Each colour is mostly the active
    tool's, but ~_OTHER_CHANCE of the time a sibling's colour and ~_INVERT_CHANCE
    of the time the inverted colour. `siblings` is the "other" colour pool - the
    caller passes it on the model tab (other models); it defaults to the other
    features in the tomogram. No-op unless a background is equipped. The continuous
    brush passes throttle=True; discrete clicks pass throttle=False. Returns True
    if shapes were spawned - callers spawning for several models in one frame use
    it to throttle only the first call and let the rest of the batch through."""
    global _last_spawn_t
    prm = _params()
    if not prm.get("enabled", False) or active_colour is None:
        return False
    now = time.monotonic()
    if throttle and (now - _last_spawn_t) < _SPAWN_MIN_DT:
        return False
    _last_spawn_t = now

    active = (float(active_colour[0]), float(active_colour[1]), float(active_colour[2]))
    style = prm.get("style", "bokeh")
    lifecycle = style == "bokeh"
    rmin = prm.get("rmin", 30.0)
    rmax = prm.get("rmax", 130.0)
    life_mul = prm.get("life_mul", 1.0)
    for _ in range(max(1, int(count))):
        rr = random.random()
        if rr < _INVERT_CHANCE:
            base = (1.0 - active[0], 1.0 - active[1], 1.0 - active[2])
        elif rr < _INVERT_CHANCE + _OTHER_CHANCE:
            sib = siblings if siblings is not None else _siblings(active)
            base = random.choice(sib) if sib else active
        else:
            base = active
        col = _hue_jitter(_soft_color(base))
        if lifecycle:
            # a gentle baseline drift in a random direction; drift is also the
            # per-blob speed threshold above which motion gets damped
            drift = random.uniform(*_DRIFT)
            ang0 = random.uniform(0.0, 2.0 * math.pi)
            vx, vy = math.cos(ang0) * drift, math.sin(ang0) * drift
        else:
            vx, vy, drift = random.uniform(-16, 16), random.uniform(-16, 16), 0.0
        _blobs.append(_Blob(
            x=random.uniform(0.0, _screen_w),
            y=random.uniform(0.0, _screen_h),
            vx=vx,
            vy=vy,
            r=random.uniform(rmin, rmax),
            color=col,
            color_target=col,
            angle=random.uniform(-0.6, 0.6),
            spin=random.uniform(-0.25, 0.25),
            breathe=random.uniform(0.15, 0.4),
            phase=random.uniform(0.0, 6.28),
            age=0.0,
            life=random.uniform(*_LIFE) * life_mul,
            drift=drift,
        ))
    if len(_blobs) > _MAX_BLOBS:
        del _blobs[: len(_blobs) - _MAX_BLOBS]
    return True


def pulse(kind: str, colour) -> None:
    # Discrete-action hook: box / copy drop a shape (unthrottled) in that colour.
    # Level-ups deliberately do NOT touch the background.
    if colour is None or kind == "levelup":
        return
    spawn(colour, throttle=False)


def notify_levelup(ev) -> None:
    # A level-up sends every live shape drifting to a NEW RANDOM hue over ~2s (the
    # transition is the colour lerp in frame()). New shapes still spawn in the
    # active colour, so the field drifts back over time.
    for b in _blobs:
        b.color_target = _soft_color(colorsys.hsv_to_rgb(
            random.random(), random.uniform(0.6, 0.9), random.uniform(0.85, 1.0)))


def _alpha(age: float, life: float) -> float:
    fade_out = _FADE_OUT_FRAC * life
    if age < _FADE_IN:
        return age / _FADE_IN
    if age > life - fade_out:
        return max(0.0, (life - age) / max(fade_out, 1e-4))
    return 1.0


def frame(dt: float, w: int, h: int, camera, cursor=None, brushing: bool = False):
    """Advance the field and return (blobs, base_colour, intensity, shape) for the
    GL renderer, or None when disabled. blobs = [(x, y, radius, rgb, angle, alpha)].
    cursor is the mouse position in screen px (bokeh discs avoid it); brushing is
    True while the LMB is held (a stronger, wider cursor push)."""
    global _screen_w, _screen_h, _cur_style, _bt
    _screen_w, _screen_h = w, h
    prm = _params()
    if not prm.get("enabled", False):
        if _blobs:
            _blobs.clear()
        return None
    style = prm.get("style", "bokeh")
    if style != _cur_style:      # switching modes: don't carry shapes across
        _blobs.clear()
        _cur_style = style
    lifecycle = style == "bokeh"
    if dt > 0.1:
        dt = 0.1
    _bt += dt

    px = camera.position[0] * camera.zoom * PARALLAX
    py = -camera.position[1] * camera.zoom * PARALLAX
    cursor_bs = (cursor[0] - px, cursor[1] - py) if (cursor is not None and lifecycle) else None
    damp = _AVOID_DAMP ** dt
    kc = 1.0 - math.exp(-dt / _TAU_RECOLOR)   # level-up hue transition (~2s)
    # while the LMB is held (brushing), the cursor pushes discs harder and farther
    avoid_r = _AVOID_R * (_AVOID_BRUSH_RANGE if brushing else 1.0)
    avoid_r2 = avoid_r * avoid_r
    avoid_accel = _AVOID_ACCEL * (_AVOID_BRUSH_STRENGTH if brushing else 1.0)

    alive: List[_Blob] = []
    for b in _blobs:
        b.age += dt
        if b.age >= b.life:
            continue
        b.color = (b.color[0] + (b.color_target[0] - b.color[0]) * kc,
                   b.color[1] + (b.color_target[1] - b.color[1]) * kc,
                   b.color[2] + (b.color_target[2] - b.color[2]) * kc)
        if cursor_bs is not None:
            dx, dy = b.x - cursor_bs[0], b.y - cursor_bs[1]
            d2 = dx * dx + dy * dy
            if 1.0 < d2 < avoid_r2:
                d = math.sqrt(d2)
                push = (1.0 - d / avoid_r) * avoid_accel
                b.vx += (dx / d) * push * dt
                b.vy += (dy / d) * push * dt
        b.x += b.vx * dt
        b.y += b.vy * dt
        if lifecycle:
            # keep a gentle baseline drift; damp only the speed ABOVE the per-blob
            # threshold (so a cursor push decays back to a calm drift, not to zero)
            speed = math.hypot(b.vx, b.vy)
            if speed > b.drift:
                scale = (b.drift + (speed - b.drift) * damp) / speed
                b.vx *= scale
                b.vy *= scale
        b.angle += b.spin * dt
        alive.append(b)
    _blobs[:] = alive

    intensity = prm.get("intensity", 0.4)
    amp = 0.08

    out = []
    for b in _blobs:
        a = _alpha(b.age, b.life)
        if a <= 0.0:
            continue
        if lifecycle:
            rad = b.r
        else:
            rad = b.r * (1.0 + amp * math.sin(_bt * b.breathe + b.phase))
        # shrink toward _FADE_OUT_SHRINK (x1/3) gradually across the fade-out
        fade_out = _FADE_OUT_FRAC * b.life
        if b.age > b.life - fade_out:
            fo = (b.life - b.age) / max(fade_out, 1e-4)   # 1 -> 0 during fade-out
            rad *= _FADE_OUT_SHRINK + (1.0 - _FADE_OUT_SHRINK) * fo
        out.append((b.x + px, b.y + py, rad, b.color, b.angle, a))
    base = tuple(cfg.COLOUR_WINDOW_BACKGROUND[:3])
    return out, base, intensity, _SHAPE.get(style, 0)


def clear() -> None:
    global _cur_style, _bt
    _blobs.clear()
    _cur_style = ""
    _bt = 0.0
