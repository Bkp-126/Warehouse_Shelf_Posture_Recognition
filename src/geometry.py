import math


def angle(a, b, c):
    u = (a[0] - b[0], a[1] - b[1])
    v = (c[0] - b[0], c[1] - b[1])
    n = math.hypot(*u) * math.hypot(*v)
    if n < 1e-8:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / n))))


def image_rect(widget_w, widget_h, image_w, image_h):
    if min(widget_w, widget_h, image_w, image_h) <= 0:
        raise ValueError("图像或控件尺寸无效")
    s = min(widget_w / image_w, widget_h / image_h)
    return ((widget_w - image_w * s) / 2, (widget_h - image_h * s) / 2, image_w * s, image_h * s)


def widget_to_normalized(x, y, ww, wh, iw, ih):
    ox, oy, w, h = image_rect(ww, wh, iw, ih)
    if not ox <= x < ox + w or not oy <= y < oy + h:
        return None
    return ((x - ox) / w, (y - oy) / h)
