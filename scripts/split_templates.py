from collections import deque
from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "public" / "templates"
OUTPUT.mkdir(parents=True, exist_ok=True)

SOURCES = [
    (Path(r"C:\Users\Lenovo\Downloads\录音转写 (1).png"), "new-year", "过年"),
    (Path(r"C:\Users\Lenovo\Downloads\录音转写 (2).png"), "school", "校服"),
    (Path(r"C:\Users\Lenovo\Downloads\录音转写 (3).png"), "fairy-tale", "童话"),
    (Path(r"C:\Users\Lenovo\Downloads\录音转写 (4).png"), "ancient", "古风"),
    (Path(r"C:\Users\Lenovo\Downloads\录音转写 (5).png"), "sports", "运动"),
]

NAMES = {
    "new-year": ["福气唐装", "灯笼红裙", "萌虎迎春", "福兔背心", "青云唐装", "梅花灯笼裙", "福字斗篷", "红包拜年装", "如意拜年装"],
    "school": ["学院蝴蝶结", "卡其小绅士", "灰裙校服", "海军蓝西装", "酒红针织衫", "披肩学院裙", "蓝衫格纹裙", "棕衫学院装", "绿背心校服"],
    "fairy-tale": ["粉红小公主", "蓝披风王子", "森林公主", "银甲骑士", "黑礼服王子", "紫星公主", "花园公主", "海军礼服", "彩虹公主"],
    "ancient": ["青玉汉服", "锦绣公子", "桃粉汉服", "花间襦裙", "山河侠客", "梅花灯笼袍", "杏黄汉服", "竹影披风", "莲花仙裙"],
    "sports": ["篮球少年", "羽毛球女孩", "足球小将", "跳绳运动装", "轮滑女孩", "网球女孩", "棒球夹克", "活力篮球装", "滑板女孩"],
}

HEAD_ADJUSTMENTS = {
    # Fur collar and side lantern shift the foreground bounding box left.
    "new-year-02": (18, 6),
    # Raised-arm composition shifts the visual body center away from the neck.
    "new-year-09": (20, 14),
}

# Templates with a fully illustrated face/hood have no usable opening for a
# real child portrait and should never enter the selectable template catalog.
EXCLUDED_TEMPLATE_IDS = {
    "new-year-03",
    "new-year-04",
}


def remove_connected_background(image):
    rgba = np.array(image.convert("RGBA"))
    if np.any(rgba[:, :, 3] < 250):
        rgba[rgba[:, :, 3] < 16, :3] = 0
        return Image.fromarray(rgba)

    rgb = rgba[:, :, :3]
    hi = rgb.max(axis=2)
    lo = rgb.min(axis=2)
    candidate = (lo > 226) & ((hi - lo) < 24)
    h, w = candidate.shape
    outside = np.zeros((h, w), dtype=bool)
    queue = deque()
    for x in range(w):
        if candidate[0, x]: queue.append((0, x))
        if candidate[h - 1, x]: queue.append((h - 1, x))
    for y in range(h):
        if candidate[y, 0]: queue.append((y, 0))
        if candidate[y, w - 1]: queue.append((y, w - 1))
    while queue:
        y, x = queue.popleft()
        if outside[y, x] or not candidate[y, x]:
            continue
        outside[y, x] = True
        if y: queue.append((y - 1, x))
        if y + 1 < h: queue.append((y + 1, x))
        if x: queue.append((y, x - 1))
        if x + 1 < w: queue.append((y, x + 1))
    rgba[outside, 3] = 0
    rgba[outside, :3] = 0
    return Image.fromarray(rgba)


def make_template(cell, slug, category, index):
    transparent = remove_connected_background(cell)
    data = np.array(transparent)
    alpha = data[:, :, 3]
    # Remove generator marks confined to the extreme bottom-right corner.
    alpha[int(alpha.shape[0] * .92):, int(alpha.shape[1] * .66):] = 0
    if index == 9:
        alpha[int(alpha.shape[0] * .90):, :] = 0
    # Accessories from an adjacent grid cell occasionally cross the cut line.
    # Remove isolated components that touch a cell edge; complete templates
    # and their intended props all have a visible margin in the source sheets.
    foreground = alpha > 20
    foreground_count = int(foreground.sum())
    seen = np.zeros_like(foreground)
    h, w = foreground.shape
    for edge_points in (
        [(y, 0) for y in range(h)], [(y, w - 1) for y in range(h)],
        [(0, x) for x in range(w)], [(h - 1, x) for x in range(w)]
    ):
        for start in edge_points:
            sy, sx = start
            if seen[sy, sx] or not foreground[sy, sx]:
                continue
            queue = deque([start])
            component = []
            while queue:
                py, px = queue.popleft()
                if seen[py, px] or not foreground[py, px]:
                    continue
                seen[py, px] = True
                component.append((py, px))
                if py: queue.append((py - 1, px))
                if py + 1 < h: queue.append((py + 1, px))
                if px: queue.append((py, px - 1))
                if px + 1 < w: queue.append((py, px + 1))
            if len(component) < foreground_count * .14:
                for py, px in component:
                    alpha[py, px] = 0
    if index == 9:
        foreground = alpha > 20
        seen = np.zeros_like(foreground)
        for sy in range(h):
            for sx in range(w):
                if seen[sy, sx] or not foreground[sy, sx]:
                    continue
                queue = deque([(sy, sx)])
                component = []
                while queue:
                    py, px = queue.popleft()
                    if seen[py, px] or not foreground[py, px]:
                        continue
                    seen[py, px] = True
                    component.append((py, px))
                    if py: queue.append((py - 1, px))
                    if py + 1 < h: queue.append((py + 1, px))
                    if px: queue.append((py, px - 1))
                    if px + 1 < w: queue.append((py, px + 1))
                if min(py for py, _ in component) > h * .80 and len(component) < foreground_count * .12:
                    for py, px in component:
                        alpha[py, px] = 0
    data[:, :, 3] = alpha
    transparent = Image.fromarray(data)

    bbox = transparent.getbbox()
    if not bbox:
        raise RuntimeError(f"No foreground found in {slug}-{index:02d}")
    foreground = transparent.crop(bbox)
    max_w, max_h = 550, 610
    scale = min(max_w / foreground.width, max_h / foreground.height)
    size = (round(foreground.width * scale), round(foreground.height * scale))
    foreground = foreground.resize(size, Image.Resampling.LANCZOS)
    x = (600 - size[0]) // 2
    y = 790 - size[1]
    canvas = Image.new("RGBA", (600, 800), (0, 0, 0, 0))
    canvas.alpha_composite(foreground, (x, y))

    # The neck is centered in each source grid cell. Map that stable grid
    # coordinate through the crop transform; raised arms no longer shift it.
    source_center_x = cell.width / 2
    neck_x = x + (source_center_x - bbox[0]) * scale
    # Most source neck openings occupy the first 8-13% of the body height.
    collar_y = y + size[1] * .11
    head_w = 215
    head_h = 225
    head_y = collar_y - head_h * .45

    filename = f"{slug}-{index:02d}.png"
    canvas.save(OUTPUT / filename, optimize=True)
    template_id = f"{slug}-{index:02d}"
    adjust_x, adjust_y = HEAD_ADJUSTMENTS.get(template_id, (0, 0))
    return {
        "id": template_id, "name": NAMES[slug][index - 1],
        "category": category, "imageUrl": f"/templates/{filename}",
        "tone": "#f1ede4", "head": {
            "x": round(max(120, min(480, neck_x + adjust_x))),
            "y": round(max(75, min(175, head_y + adjust_y))), "w": head_w, "h": head_h
        }
    }


def main():
    catalog = []
    for old_template in OUTPUT.glob("*.png"):
        old_template.unlink()
    for path, slug, category in SOURCES:
        source = Image.open(path)
        cell_w, cell_h = source.width / 3, source.height / 3
        for row in range(3):
            for col in range(3):
                left, top = round(col * cell_w), round(row * cell_h)
                right, bottom = round((col + 1) * cell_w), round((row + 1) * cell_h)
                cell = source.crop((left, top, right, bottom))
                index = row * 3 + col + 1
                template_id = f"{slug}-{index:02d}"
                if template_id not in EXCLUDED_TEMPLATE_IDS:
                    catalog.append(make_template(cell, slug, category, index))

    catalog_text = json.dumps(catalog, ensure_ascii=False, indent=2)
    (OUTPUT / "catalog.json").write_text(catalog_text, encoding="utf-8")
    (ROOT / "src" / "template-catalog.json").write_text(catalog_text, encoding="utf-8")

    sheet = Image.new("RGB", (5 * 220, 9 * 300), "#e8e7e1")
    for i, item in enumerate(catalog):
        thumb = Image.open(OUTPUT / Path(item["imageUrl"]).name)
        thumb.thumbnail((200, 250), Image.Resampling.LANCZOS)
        col, row = i // 9, i % 9
        tile = Image.new("RGBA", (200, 270), "white")
        tile.alpha_composite(thumb, ((200 - thumb.width) // 2, 0))
        ImageDraw.Draw(tile).text((8, 252), item["name"], fill="#222")
        sheet.paste(tile.convert("RGB"), (col * 220 + 10, row * 300 + 10))
    sheet.save(OUTPUT / "contact-sheet.jpg", quality=90)
    print(f"Created {len(catalog)} templates in {OUTPUT}")


if __name__ == "__main__":
    main()
