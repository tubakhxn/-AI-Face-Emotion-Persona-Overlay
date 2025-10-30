"""
overlay_utils.py
Drawing utilities for neon cyberpunk HUD overlays: glow rounded rectangle, scanline,
glitch text, FPS, and screenshot helpers.

Build by Tuba khan
"""
import cv2
import numpy as np
import time
import os
from typing import Tuple

# Neon color palette (cyberpunk blue-focused)
NEON_CYAN = (200, 220, 255)  # soft cyan (BGR)
NEON_BLUE = (160, 90, 255)   # deeper blue/purple for accents
NEON_ACCENT = (180, 140, 255)


def draw_rounded_rect(img, rect, color, thickness=2, radius=20, glow=False):
    """Draw a rounded rectangle with optional glow.

    rect: (x,y,w,h)
    """
    x, y, w, h = rect
    overlay = img.copy()
    # Draw main rounded rectangle by drawing 4 lines and 4 circles
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    # lines
    cv2.line(overlay, (x1+radius, y1), (x2-radius, y1), color, thickness)
    cv2.line(overlay, (x1+radius, y2), (x2-radius, y2), color, thickness)
    cv2.line(overlay, (x1, y1+radius), (x1, y2-radius), color, thickness)
    cv2.line(overlay, (x2, y1+radius), (x2, y2-radius), color, thickness)
    # corners
    cv2.ellipse(overlay, (x1+radius, y1+radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x2-radius, y1+radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x1+radius, y2-radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(overlay, (x2-radius, y2-radius), (radius, radius), 0, 0, 90, color, thickness)

    if glow:
        # create glow by adding blurred layers with cyan/tint
        mask = np.zeros_like(img)
        cv2.rectangle(mask, (x1, y1), (x2, y2), color, -1)
        for k,alpha in ((25,0.12),(15,0.08),(7,0.04)):
            b = cv2.GaussianBlur(mask, (k*2+1, k*2+1), 0)
            img[:] = cv2.addWeighted(img, 1.0, b, alpha, 0)

    # merge overlay
    img[:] = cv2.addWeighted(img, 1.0, overlay, 0.9, 0)
    return img


def draw_scanline(img, y_pos: int, color=None, thickness=2):
    h, w = img.shape[:2]
    overlay = img.copy()
    # horizontal glowing line with gradient
    if color is None:
        color = NEON_CYAN
    cv2.line(overlay, (0, y_pos), (w, y_pos), color, thickness)
    # add a faint thicker line
    cv2.line(overlay, (0, y_pos-2), (w, y_pos-2), (max(0,color[0]//2), max(0,color[1]//2), max(0,color[2]//2)), thickness+2)
    img[:] = cv2.addWeighted(img, 0.9, overlay, 0.4, 0)
    return img


def draw_fps(img, fps: float, pos=(10, 28)):
    text = f"FPS: {int(fps)}"
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, NEON_BLUE, 2, cv2.LINE_AA)


def draw_emotion_label(img, label: str, conf: float, persona: str, bbox: Tuple[int,int,int,int], alpha: float = 1.0):
    x, y, w, h = bbox
    # position above the bbox
    px, py = x, max(10, y-12)
    text = f"{label} ({int(conf*100)}%)"
    # glitch effect: small random offsets
    jitter_x = int(4 * (1-alpha))
    offset = (jitter_x, 0)
    # draw shadow
    cv2.putText(img, text, (px+2+offset[0], py+2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10,10,10), 3, cv2.LINE_AA)
    # main neon
    color = NEON_CYAN if alpha>0.5 else NEON_BLUE
    cv2.putText(img, text, (px+offset[0], py), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    # persona below
    persona_pos = (px, py + 22)
    cv2.putText(img, persona, persona_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, NEON_CYAN, 1, cv2.LINE_AA)


def draw_glitch_text(img, text: str, pos=(30, 60), base_color=(220, 90, 255)):
    # draw multi-colored offset layers to emulate glitch
    x, y = pos
    cv2.putText(img, text, (x-2, y), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,0,0), 6, cv2.LINE_AA)
    cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_DUPLEX, 0.9, (255,0,100), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x-1, y-1), cv2.FONT_HERSHEY_DUPLEX, 0.9, (100,200,255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.9, base_color, 2, cv2.LINE_AA)


def draw_status_panel(img, lines, pos=(10, 60), bg_color=(10,10,20), alpha=0.6):
    """Draw a small translucent status panel with given text lines."""
    x, y = pos
    h, w = img.shape[:2]
    panel_w = 320
    panel_h = 20 + 18 * len(lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+panel_w, y+panel_h), bg_color, -1)
    img[:] = cv2.addWeighted(img, 1.0, overlay, alpha, 0)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x+8, y+18 + i*18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, NEON_CYAN, 1, cv2.LINE_AA)
    return img


def save_screenshot(img, out_dir='screenshots') -> str:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    ts = int(time.time())
    fname = os.path.join(out_dir, f'screenshot_{ts}.png')
    cv2.imwrite(fname, img)
    return fname


if __name__ == "__main__":
    # quick visual demo when run as script
    cap = cv2.VideoCapture(0)
    start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h,w = frame.shape[:2]
        # demo rounded rect in center
        rect = (w//4, h//4, w//2, h//2)
        draw_rounded_rect(frame, rect, NEON_CYAN, thickness=2, glow=True)
        y = int((time.time()-start)*150) % h
        draw_scanline(frame, y)
        draw_glitch_text(frame, 'AI FACE PERSONA', pos=(30,50))
        draw_fps(frame, 30)
        cv2.imshow('hud demo', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
