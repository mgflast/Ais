# ImGui helpers: XP HUD, profile panel, the centre level-up celebration, and
# the perk-gated cursor/box particle emitters. The editor's loop drives these.
from __future__ import annotations

import colorsys
import math
import os
import random
import time
from typing import List, Optional, Tuple

import imgui

import Ais.core.config as cfg
from . import background
from . import cosmetics
from . import events
from . import orbs
from . import particles
from . import perks
from . import profile as _profile


Color = Tuple[float, float, float]


# Skill level at which the visual "perks" (cursor sparkles, etc.) unlock.
PERK_UNLOCK_LEVEL = 5

# 8-direction outline offsets for big-text rendering.
_OUTLINE_OFFSETS_8 = ((-2, -2), (0, -2), (2, -2), (-2, 0), (2, 0), (-2, 2), (0, 2), (2, 2))

# Editor passes the panel logo texture handle on startup.
_panel_icon_renderer_id: Optional[int] = None

# Large font baked by the editor for crisp level-up text; None -> upscale the
# default font (blurry fallback). See big_font_path().
_big_font = None
BIG_FONT_PX = 64


def set_panel_icon(renderer_id: int) -> None:
    global _panel_icon_renderer_id
    _panel_icon_renderer_id = renderer_id


def set_big_font(font) -> None:
    global _big_font
    _big_font = font


def big_font_path() -> Optional[str]:
    # matplotlib bundles DejaVuSans on every platform; fall back to a system font
    try:
        import matplotlib
        p = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    for p in ("C:/Windows/Fonts/arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf"):
        if os.path.exists(p):
            return p
    return None


_active_levelup: Optional[events.LevelUp] = None
_levelup_started_at: float = 0.0
LEVELUP_DURATION_S = 5.2

# Firework pops queued at (fire_time, x, y, color, n) and drained each frame so
# they go off in sequence rather than all at once.
_pending_bursts: List[Tuple[float, float, float, Color, int]] = []

# Level-up confetti bursts queued at (fire_time, x, y, n, color); drained each
# frame so the celebration hops between locations over a couple of seconds.
_pending_confetti: List[Tuple[float, float, float, int, Color, bool]] = []

# Last known cursor position, captured each frame; used for cursor confetti.
_cursor_pos: Tuple[float, float] = (0.0, 0.0)


def set_cursor_pos(x: float, y: float) -> None:
    global _cursor_pos
    _cursor_pos = (float(x), float(y))

_skills_open: bool = False

# Effect on/off toggles, surfaced in the editor's Party-mode menu.
EFFECT_TOGGLES = (
    ("PERK_XP_HUD",     "Level overlay (top-right)"),
    ("PERK_CURSOR",     "Cursor sparkles"),
    ("PERK_XP_ORBS",    "XP orbs"),
    ("PERK_BOX_BURST",  "Box-placement burst"),
    ("PERK_CONFETTI",   "Level-up confetti"),
    ("PERK_MILESTONE",  "Level-up celebration"),
    ("PERK_COLOR_ANIM", "Colour animations"),
)


def toggle_skills_panel() -> None:
    global _skills_open
    _skills_open = not _skills_open


def is_skills_open() -> bool:
    return _skills_open


def tick_particles(dt: float) -> None:
    particles.tick(dt)
    orbs.tick(dt)


def draw_particles(camera=None, screen_h: float = 0.0) -> None:
    particles.draw(camera, screen_h)
    orbs.draw()


def emit_brush_trail(cx: float, cy: float, radius_px: float, color: Color, skill_level: int) -> None:
    if not cfg.settings.get("PERK_CURSOR", True):
        return
    tier = perks.perk_for_level(skill_level)
    if tier is None or tier.cursor_n <= 0:
        return
    prm = cosmetics.params(cosmetics.CURSOR)
    particles.emit_brush_ring(cx, cy, radius_px, color, n=tier.cursor_n, h_amp=tier.hue_amp, world=True,
                              palette=prm.get("palette", "feature"), size_mul=prm.get("size_mul", 1.0))


def emit_box_burst(cx: float, cy: float, size_px: float, color: Color, skill_level: int) -> None:
    if not cfg.settings.get("PERK_BOX_BURST", True):
        return
    tier = perks.perk_for_level(skill_level)
    if tier is None or tier.box_burst_n <= 0:
        return
    prm = cosmetics.params(cosmetics.BURST)
    particles.emit_box_outline_burst(cx, cy, size_px, color, n=max(1, tier.box_burst_n // 2), h_amp=tier.hue_amp, world=True,
                                     palette=prm.get("palette", "feature"), size_mul=prm.get("size_mul", 1.0))


def _ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _luminance(c: Color) -> float:
    return 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]


def _readable_on(bg: Color) -> Tuple[Color, Color]:
    if _luminance(bg) > 0.55:
        return ((0.06, 0.06, 0.10), (1.0, 1.0, 1.0))
    return ((0.97, 0.97, 0.98), (0.0, 0.0, 0.0))


def _draw_text_outlined(dl, x: float, y: float, text: str, fg: Color, outline: Color, outline_alpha: float = 0.7) -> None:
    out_u = imgui.get_color_u32_rgba(outline[0], outline[1], outline[2], outline_alpha)
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        dl.add_text(x + dx, y + dy, out_u, text)
    dl.add_text(x, y, imgui.get_color_u32_rgba(fg[0], fg[1], fg[2], 1.0), text)


def _tinted_badge_bg(base: Color, mix: float = 0.55) -> Color:
    return (
        _lerp(0.86, base[0], mix),
        _lerp(0.86, base[1], mix),
        _lerp(0.78, base[2], mix),
    )


def _draw_swatch(dl, x: float, y: float, size: float, color: Color) -> None:
    dl.add_rect_filled(x, y, x + size, y + size, imgui.get_color_u32_rgba(color[0], color[1], color[2], 1.0), size * 0.30)
    dl.add_rect(x, y, x + size, y + size, imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 0.30), size * 0.30, 0, 1.0)


def _draw_track(
    dl, x: float, y: float, w: float, h: float, frac: float,
    bg: Color, fill: Color, overlay: str = "",
) -> None:
    dl.add_rect_filled(x, y, x + w, y + h, imgui.get_color_u32_rgba(bg[0], bg[1], bg[2], 1.0), h * 0.5)
    fw = max(h, w * max(0.0, min(1.0, frac)))
    if fw > 0:
        dl.add_rect_filled(x, y, x + fw, y + h, imgui.get_color_u32_rgba(fill[0], fill[1], fill[2], 1.0), h * 0.5)
    if overlay:
        tw, th = imgui.calc_text_size(overlay)
        tx = x + (w - tw) * 0.5
        ty = y + (h - th) * 0.5 - 1
        blend_color = fill if frac >= 0.45 else bg
        fg, ol = _readable_on(blend_color)
        _draw_text_outlined(dl, tx, ty, overlay, fg, ol, outline_alpha=0.8)


def _truncate_to_width(text: str, max_w: float) -> str:
    tw, _ = imgui.calc_text_size(text)
    if tw <= max_w:
        return text
    ellip = "..."
    while text and imgui.calc_text_size(text + ellip)[0] > max_w:
        text = text[:-1]
    return text + ellip if text else ""


_XP_HUD_W = 280
_XP_HUD_ROW_H = 28
_XP_HUD_RECENT_WINDOW_S = 60.0
_XP_HUD_MAX_ROWS = 10


def render_xp_hud(window_width: int, window_height: int, hidden: bool = False) -> None:
    if hidden or not cfg.settings.get("PERK_XP_HUD", True):
        return

    p = _profile.get_profile()

    # Keyed by canonical skill name, matching events/profile identity.
    active_names: List[str] = []
    color_overrides: dict = {}
    if cfg.se_active_frame is not None:
        seen = set()
        for f in cfg.se_active_frame.features:
            ck = _profile.canonical(f.title)
            if ck in seen:
                continue
            seen.add(ck)
            active_names.append(ck)
            color_overrides[ck] = tuple(f.colour)

    recent = events.recent_features(_XP_HUD_RECENT_WINDOW_S)

    # Only show features that recently gained XP; features that are merely present
    # in the frame (existing annotations, no recent activity) are not listed.
    rows: List[Tuple[str, float, bool]] = []
    for name, dt in recent:
        if p.is_hidden(name):
            continue
        rows.append((name, dt, name in active_names))

    rows = rows[:_XP_HUD_MAX_ROWS]
    if not rows:
        return

    n = len(rows)
    body_h = n * _XP_HUD_ROW_H + 8
    pad = 14
    win_x = window_width - _XP_HUD_W - pad
    win_y = pad + 22

    flags = (
        imgui.WINDOW_NO_TITLE_BAR
        | imgui.WINDOW_NO_RESIZE
        | imgui.WINDOW_NO_MOVE
        | imgui.WINDOW_NO_SCROLLBAR
        | imgui.WINDOW_NO_SAVED_SETTINGS
        | imgui.WINDOW_NO_FOCUS_ON_APPEARING
        | imgui.WINDOW_NO_NAV
        | imgui.WINDOW_NO_INPUTS
        | imgui.WINDOW_NO_BACKGROUND
    )

    imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0, 0))
    imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 0.0)

    imgui.set_next_window_position(win_x, win_y, imgui.ALWAYS)
    imgui.set_next_window_size(_XP_HUD_W, body_h)
    imgui.begin("##progression_xp_hud", False, flags)

    dl = imgui.get_window_draw_list()
    win_pos = imgui.get_window_position()

    track_col = (0.18, 0.18, 0.20)   # slate HUD look (fixed)
    track_alpha = 0.9
    backdrop_a = 0.55
    orb_targets = {}

    for i, (name, dt, in_frame) in enumerate(rows):
        row_top = win_pos[1] + 4 + i * _XP_HUD_ROW_H
        row_x = win_pos[0]

        # Fade the row out toward the end of the recent-activity window. While a
        # feature is actively annotated its dt keeps resetting, so it stays at full
        # opacity; it only fades once XP gains stop.
        fade_start = _XP_HUD_RECENT_WINDOW_S - 3.0
        row_alpha = 1.0 if dt < fade_start else max(0.0, 1.0 - (dt - fade_start) / 3.0)
        if row_alpha <= 0.0:
            continue

        gain_pulse = max(0.0, 1.0 - dt / 1.2) if dt < 1.2 else 0.0

        color = color_overrides.get(name) or p.skill_color(name)
        xp = p.skill_xp(name)
        L, into, needed = _profile.xp_into_level(xp)

        content_x = row_x
        content_right = row_x + _XP_HUD_W - 8

        lv_text = f"level {L}"
        lv_tw, lv_th = imgui.calc_text_size(lv_text)
        lv_x = content_right - lv_tw

        name_x = content_x
        name_max_w = lv_x - name_x - 8
        name_text = _truncate_to_width(p.display_name(name), name_max_w)
        name_fg = (0.0, 0.0, 0.0)
        if gain_pulse > 0:
            name_fg = (
                _lerp(0.0, color[0], 0.7 * gain_pulse),
                _lerp(0.0, color[1], 0.7 * gain_pulse),
                _lerp(0.0, color[2], 0.7 * gain_pulse),
            )
        _draw_text_outlined(dl, name_x, row_top + 3, name_text, name_fg, (1.0, 1.0, 1.0), outline_alpha=0.85 * row_alpha)
        _draw_text_outlined(dl, lv_x, row_top + 3, lv_text, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), outline_alpha=0.85 * row_alpha)

        # progress bar: square, themed track over a black backdrop for contrast
        bar_x = content_x
        bar_y = row_top + 20
        bar_w = content_right - bar_x
        bar_h = 6.0
        frac = (into / needed) if needed else 1.0
        # orbs fly to where the bar's progress currently reaches, not its start
        _progress_x = bar_x + max(bar_h, bar_w * max(0.0, min(1.0, frac)))
        orb_targets[name] = (_progress_x, bar_y + bar_h * 0.5)
        flash = max(gain_pulse, orbs.impact_pulse(name))   # brighten on gain and on orb landing
        dl.add_rect_filled(bar_x - 1, bar_y - 1, bar_x + bar_w + 1, bar_y + bar_h + 1,
                           imgui.get_color_u32_rgba(0.0, 0.0, 0.0, backdrop_a * row_alpha), 0.0)
        dl.add_rect_filled(bar_x, bar_y, bar_x + bar_w, bar_y + bar_h,
                           imgui.get_color_u32_rgba(track_col[0], track_col[1], track_col[2], track_alpha * row_alpha), 0.0)
        fill_w = max(bar_h, bar_w * max(0.0, min(1.0, frac)))
        if fill_w > 0:
            fill = color
            if flash > 0:
                fill = (
                    _lerp(color[0], 1.0, 0.40 * flash),
                    _lerp(color[1], 1.0, 0.40 * flash),
                    _lerp(color[2], 1.0, 0.40 * flash),
                )
            dl.add_rect_filled(bar_x, bar_y, bar_x + fill_w, bar_y + bar_h,
                               imgui.get_color_u32_rgba(fill[0], fill[1], fill[2], row_alpha), 0.0)
        if flash > 0:
            dl.add_rect_filled(bar_x, bar_y, bar_x + bar_w, bar_y + bar_h,
                               imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.30 * flash), 0.0)

    orbs.set_targets(orb_targets)
    imgui.end()
    imgui.pop_style_var(2)


_LEVELUP_CONFETTI_N = 7           # main staggered bursts
_LEVELUP_CONFETTI_STEP = 0.45     # seconds between main bursts
_LEVELUP_QUICK_N = 3              # extra bursts fired right at the start
_LEVELUP_QUICK_STEP = 0.08        # in quick succession


def _schedule_levelup_confetti(ev, now: float, window_width: int, window_height: int) -> None:
    # Queue a run of confetti bursts: a quick opening flurry (_LEVELUP_QUICK_N in
    # rapid succession) followed by the main staggered sequence. Most land at
    # random on-screen positions; every 2nd-3rd one goes off at the cursor
    # (resolved when it fires, so it tracks the moving cursor); and roughly 1 in 4
    # is all-colour confetti (color=None) rather than the feature's colour.
    times = [i * _LEVELUP_QUICK_STEP for i in range(_LEVELUP_QUICK_N)]
    times += [i * _LEVELUP_CONFETTI_STEP for i in range(_LEVELUP_CONFETTI_N)]
    next_cursor = random.choice((1, 2))
    for i, t in enumerate(times):
        at_cursor = (i == next_cursor)
        if at_cursor:
            next_cursor += random.choice((2, 3))
            x = y = 0.0   # resolved to the live cursor position at fire time
        else:
            x = random.uniform(window_width * 0.12, window_width * 0.90)
            y = random.uniform(window_height * 0.12, window_height * 0.62)
        color = None if random.random() < 0.25 else ev.color
        n = random.randint(28, 44)
        _pending_confetti.append((now + t, x, y, n, color, at_cursor))


def render_level_up(window_width: int, window_height: int, hidden: bool = False) -> None:
    global _active_levelup, _levelup_started_at
    if hidden:
        return
    now = time.time()
    # fire any firework pops whose scheduled time has arrived
    if _pending_bursts:
        for (_t, _bx, _by, _bc, _bn) in [b for b in _pending_bursts if b[0] <= now]:
            particles.emit_burst(_bx, _by, _bc, _bn)
        _pending_bursts[:] = [b for b in _pending_bursts if b[0] > now]
    # fire any staggered level-up confetti bursts whose time has arrived
    if _pending_confetti:
        for (_t, _x, _y, _n, _c, _cur) in [b for b in _pending_confetti if b[0] <= now]:
            px, py = _cursor_pos if _cur else (_x, _y)
            particles.emit_confetti_burst(px, py, n=_n, color=_c)
        _pending_confetti[:] = [b for b in _pending_confetti if b[0] > now]
    if _active_levelup is None:
        ev = events.pop_level_up()
        if ev is None:
            return
        background.notify_levelup(ev)   # whole-field colour swell in the background
        _cx, _cy = window_width * 0.5, window_height * 0.5
        if cfg.settings.get("PERK_CONFETTI", True):
            # confetti bursts outward from the centre LEVEL text and falls, like
            # the party-mode-on pop (feature-coloured rather than multicolour).
            particles.emit_confetti_burst(_cx, _cy, n=110, color=ev.color)
            _schedule_levelup_confetti(ev, now, window_width, window_height)
        # confetti fires regardless; the centre text is opt-out via PERK_MILESTONE
        if not cfg.settings.get("PERK_MILESTONE", True):
            return
        _active_levelup = ev
        _levelup_started_at = now
        # a couple of extra spark pops around the text, ~140 ms apart
        if cfg.settings.get("PERK_CONFETTI", True):
            _plan = ((0.0, 0.0, 40), (-0.5, 0.04, 26), (0.5, 0.04, 26))
            _spread = window_width * 0.10
            for _i, (_ox, _oy, _bn) in enumerate(_plan):
                _pending_bursts.append((now + _i * 0.14,
                                        _cx + _spread * _ox,
                                        _cy + window_height * _oy,
                                        ev.color, _bn))

    elapsed = now - _levelup_started_at
    if elapsed > LEVELUP_DURATION_S:
        _active_levelup = None
        return

    ev = _active_levelup

    # phases:
    # 0.00-0.45: scale 0 → 1.25 (overshoot)  with elastic-ish curve, alpha 0 → 1
    # 0.45-4.20: hold at scale 1.0, alpha 1
    # 4.20-5.20: alpha 1 → 0, scale 1.0 → 1.10
    if elapsed < 0.45:
        t = elapsed / 0.45
        if t < 0.7:
            scale = 1.25 * _ease_out_cubic(t / 0.7)
        else:
            u = (t - 0.7) / 0.3
            scale = 1.25 - 0.25 * _ease_out_cubic(u)
        alpha = min(1.0, t * 2.0)
    elif elapsed < 4.20:
        scale = 1.0
        alpha = 1.0
    else:
        t = (elapsed - 4.20) / 1.00
        scale = 1.0 + 0.10 * t
        alpha = max(0.0, 1.0 - t)

    alpha = max(0.0, min(1.0, alpha))

    text = f"LEVEL {ev.new_level}"
    use_big = _big_font is not None
    if use_big:
        imgui.push_font(_big_font)
    # imgui asserts scale > 0 (first frame is 0 from the ease-in). With the baked
    # font cur_scale rests at 1.0 for crisp 1:1 text; else upscale the default.
    if use_big:
        cur_scale = max(0.01, scale)
        sub_scale = 0.42
    else:
        cur_scale = max(0.01, 4.2 * scale)
        sub_scale = 1.6

    base_tw, base_th = imgui.calc_text_size(text)
    big_tw = base_tw * cur_scale
    big_th = base_th * cur_scale

    # fit the feature name to the window: keep the full name whenever it fits,
    # and only truncate if it would overflow the screen width.
    max_sub_w = max(120.0, window_width - 100.0)
    if imgui.calc_text_size(ev.name)[0] * sub_scale <= max_sub_w:
        sub_text = ev.name
    else:
        sub_text = _truncate_to_width(ev.name, max_sub_w / sub_scale)
    sub_base_tw, sub_base_th = imgui.calc_text_size(sub_text)
    sub_tw = sub_base_tw * sub_scale
    sub_th = sub_base_th * sub_scale

    win_w = max(big_tw, sub_tw) + 80
    win_h = big_th + sub_th + 40
    win_x = (window_width - win_w) / 2
    win_y = (window_height - win_h) / 2

    flags = (
        imgui.WINDOW_NO_TITLE_BAR
        | imgui.WINDOW_NO_RESIZE
        | imgui.WINDOW_NO_MOVE
        | imgui.WINDOW_NO_SCROLLBAR
        | imgui.WINDOW_NO_SAVED_SETTINGS
        | imgui.WINDOW_NO_FOCUS_ON_APPEARING
        | imgui.WINDOW_NO_NAV
        | imgui.WINDOW_NO_INPUTS
        | imgui.WINDOW_NO_BACKGROUND
    )
    imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (0, 0))
    imgui.push_style_var(imgui.STYLE_WINDOW_BORDERSIZE, 0.0)
    imgui.set_next_window_position(int(win_x), int(win_y), imgui.ALWAYS)
    imgui.set_next_window_size(int(win_w), int(win_h))
    imgui.begin("##progression_milestone", False, flags)

    imgui.set_window_font_scale(cur_scale)
    text_offset_x = (win_w - big_tw) / 2
    text_offset_y = 16
    for ox, oy in _OUTLINE_OFFSETS_8:
        imgui.set_cursor_pos((text_offset_x + ox, text_offset_y + oy))
        imgui.push_style_color(imgui.COLOR_TEXT, 0.0, 0.0, 0.0, alpha * 0.55)
        imgui.text(text)
        imgui.pop_style_color(1)
    halo_color = (
        min(1.0, ev.color[0] + 0.4),
        min(1.0, ev.color[1] + 0.4),
        min(1.0, ev.color[2] + 0.4),
    )
    for ox, oy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        imgui.set_cursor_pos((text_offset_x + ox, text_offset_y + oy))
        imgui.push_style_color(imgui.COLOR_TEXT, halo_color[0], halo_color[1], halo_color[2], alpha * 0.35)
        imgui.text(text)
        imgui.pop_style_color(1)
    imgui.set_cursor_pos((text_offset_x, text_offset_y))
    imgui.push_style_color(imgui.COLOR_TEXT, ev.color[0], ev.color[1], ev.color[2], alpha)
    imgui.text(text)
    imgui.pop_style_color(1)

    imgui.set_window_font_scale(sub_scale)
    sub_offset_x = (win_w - sub_tw) / 2
    sub_offset_y = text_offset_y + big_th + 8
    for ox, oy in ((1, 1), (-1, -1)):
        imgui.set_cursor_pos((sub_offset_x + ox, sub_offset_y + oy))
        imgui.push_style_color(imgui.COLOR_TEXT, 0.0, 0.0, 0.0, alpha * 0.5)
        imgui.text(sub_text)
        imgui.pop_style_color(1)
    imgui.set_cursor_pos((sub_offset_x, sub_offset_y))
    imgui.push_style_color(imgui.COLOR_TEXT, ev.color[0], ev.color[1], ev.color[2], alpha)
    imgui.text(sub_text)
    imgui.pop_style_color(1)

    imgui.set_window_font_scale(1.0)
    imgui.end()
    imgui.pop_style_var(2)
    if use_big:
        imgui.pop_font()


def _begin_panel(title: str, tag: str, w: int, h: int):
    imgui.set_next_window_size(w, h, imgui.FIRST_USE_EVER)
    imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (16, 14))
    imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, cfg.WINDOW_ROUNDING)
    imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, *cfg.COLOUR_WINDOW_BACKGROUND)
    imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, *cfg.COLOUR_TITLE_BACKGROUND)
    imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, *cfg.COLOUR_TITLE_BACKGROUND)
    imgui.push_style_color(imgui.COLOR_BORDER, *cfg.COLOUR_FRAME_EXTRA_DARK[:3], 0.85)
    imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_TEXT)
    return imgui.begin(f"{title}##{tag}", True, imgui.WINDOW_NO_COLLAPSE)


def _end_panel() -> None:
    imgui.end()
    imgui.pop_style_color(5)
    imgui.pop_style_var(2)


def render_skills_panel() -> None:
    global _skills_open
    if not _skills_open:
        return
    expanded, opened = _begin_panel("Levels", "progression_skills", 620, 540)
    if not opened:
        _skills_open = False
    if not expanded:
        _end_panel()
        return

    p = _profile.get_profile()
    _render_header_card(p)
    imgui.dummy(0, 12)
    _section_label("SKILLS")
    visible = p.visible_skills()
    if not visible:
        imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_TEXT_DISABLED)
        imgui.text("  Annotate a feature to begin a skill.")
        imgui.pop_style_color(1)
    else:
        _render_skill_rows(visible, p)
    _end_panel()


_SKILL_ROW_H = 46
_SKILL_COLS = 2
_SKILL_COL_GAP = 16


def _readable_accent(color, dark: bool):
    # A version of the feature colour that reads on the current theme: lightened
    # toward white on dark backgrounds, darkened on light ones.
    if dark:
        return tuple(c + (1.0 - c) * 0.45 for c in color)
    return tuple(c * 0.5 for c in color)


def _render_skill_rows(visible: dict, p: _profile.Profile) -> None:
    dl = imgui.get_window_draw_list()
    avail_w = imgui.get_content_region_available_width()
    items = sorted(visible.items(), key=lambda kv: (-_profile.level_for_xp(kv[1]), kv[0]))
    mx, my = imgui.get_mouse_pos()
    dark = cfg.settings.get("DARK_MODE", False)
    track_bg = (0.32, 0.32, 0.35) if dark else (0.80, 0.80, 0.74)

    cols = _SKILL_COLS
    col_w = (avail_w - _SKILL_COL_GAP * (cols - 1)) / cols
    base_x, base_y = imgui.get_cursor_screen_pos()
    row_pitch = _SKILL_ROW_H + 4
    n_rows = (len(items) + cols - 1) // cols

    for i, (name, xp) in enumerate(items):
        col = i % cols
        x0 = base_x + col * (col_w + _SKILL_COL_GAP)
        y0 = base_y + (i // cols) * row_pitch
        L, into, needed = _profile.xp_into_level(xp)
        color = p.skill_color(name)

        if (x0 <= mx <= x0 + col_w) and (y0 <= my <= y0 + _SKILL_ROW_H):
            dl.add_rect_filled(x0 - 4, y0, x0 + col_w + 4, y0 + _SKILL_ROW_H,
                               imgui.get_color_u32_rgba(*cfg.COLOUR_TITLE_BACKGROUND[:3], 0.6), 6.0)

        chip = 30.0
        cx, cy = x0 + 2, y0 + (_SKILL_ROW_H - chip) * 0.5
        dl.add_rect_filled(cx, cy, cx + chip, cy + chip,
                           imgui.get_color_u32_rgba(color[0], color[1], color[2], 1.0), 6.0)
        dl.add_rect(cx, cy, cx + chip, cy + chip,
                    imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 0.30), 6.0, 0, 1.0)

        text_x = cx + chip + 10
        lv = f"Lv {L}"
        lv_tw, _ = imgui.calc_text_size(lv)
        lv_x = x0 + col_w - lv_tw - 20
        name_text = _truncate_to_width(p.display_name(name), lv_x - text_x - 6)
        dl.add_text(text_x, y0 + 5, imgui.get_color_u32_rgba(*cfg.COLOUR_TEXT[:3], 1.0), name_text)
        lvl_col = _readable_accent(color, dark)
        dl.add_text(lv_x, y0 + 5, imgui.get_color_u32_rgba(lvl_col[0], lvl_col[1], lvl_col[2], 1.0), lv)

        bar_x = text_x
        bar_y = y0 + 26
        bar_w = x0 + col_w - bar_x - 20
        bar_h = 8.0
        if needed == 0:
            frac, overlay = 1.0, "MAX"
        else:
            frac, overlay = into / needed, f"{into:,} / {needed:,}"
        _draw_track(dl, bar_x, bar_y, bar_w, bar_h, frac, bg=track_bg, fill=color, overlay=overlay)

        imgui.set_cursor_screen_pos((x0 + col_w - 16, y0 + 3))
        if imgui.invisible_button(f"##hide_{name}", 14, 14):
            p.hide(name)
            _profile.mark_dirty()
        if imgui.is_item_hovered():
            with imgui.begin_tooltip():
                imgui.text("Hide this skill")
        g_tw, g_th = imgui.calc_text_size("x")
        dl.add_text(x0 + col_w - 16 + (14 - g_tw) * 0.5, y0 + 3 + (14 - g_th) * 0.5,
                    imgui.get_color_u32_rgba(0.50, 0.50, 0.54, 0.9), "x")

    imgui.set_cursor_screen_pos((base_x, base_y))
    imgui.dummy(avail_w, n_rows * row_pitch + 4)


def _render_header_card(p: _profile.Profile) -> None:
    dl = imgui.get_window_draw_list()
    avail_w = imgui.get_content_region_available_width()
    x0, y0 = imgui.get_cursor_screen_pos()

    sprite = 60.0
    pad_top = 10.0
    gap = 4.0
    overall = p.overall_level()
    big = f"Total Level  {overall}"
    big_tw, big_th = imgui.calc_text_size(big)
    card_h = pad_top + sprite + gap + big_th + 12.0

    dl.add_rect_filled(
        x0, y0, x0 + avail_w, y0 + card_h,
        imgui.get_color_u32_rgba(*cfg.COLOUR_TITLE_BACKGROUND[:3], 1.0),
        cfg.WINDOW_ROUNDING,
    )
    dl.add_rect(
        x0, y0, x0 + avail_w, y0 + card_h,
        imgui.get_color_u32_rgba(*cfg.COLOUR_FRAME_EXTRA_DARK[:3], 1.0),
        cfg.WINDOW_ROUNDING, 0, 1.0,
    )

    # the ais boot sprite, centred and a bit small
    if _panel_icon_renderer_id is not None:
        sx = x0 + (avail_w - sprite) * 0.5
        sy = y0 + pad_top
        dl.add_image(
            _panel_icon_renderer_id,
            (sx, sy), (sx + sprite, sy + sprite),
            (0.0, 0.0), (1.0, 1.0),
            imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0),
        )

    # total level, centred below the sprite, in the theme text colour (readable in dark mode)
    dl.add_text(
        x0 + (avail_w - big_tw) * 0.5, y0 + pad_top + sprite + gap,
        imgui.get_color_u32_rgba(*cfg.COLOUR_TEXT[:3], 1.0), big,
    )

    imgui.dummy(avail_w, card_h)


def _section_label(label: str) -> None:
    imgui.push_style_color(imgui.COLOR_TEXT, *cfg.COLOUR_TEXT_DISABLED)
    imgui.text(label)
    imgui.pop_style_color(1)
    imgui.separator()
    imgui.spacing()


