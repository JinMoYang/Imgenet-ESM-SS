import os
import json
import cv2
import numpy as np
import random
import math

# ====== 경로 설정 ======
root_dir = "/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_outer/batch_test/batch_test_phase2_validation/merged_annotations_with_images"
save_root = "/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_outer/batch_test/batch_test_phase2_validation/merged_annotations_with_images/visualized"
os.makedirs(save_root, exist_ok=True)

# ====== 유틸 ======
def safe_str(v):
    if isinstance(v, str): return v.strip()
    if v is None: return ""
    return str(v).strip()

def generate_distinct_colors(keys):
    """HSV → BGR 색상표 생성 (OpenCV 내부 일관성 유지)"""
    n = len(keys)
    if n == 0: return {}
    hues = np.linspace(0, 179, n, endpoint=False)
    random.shuffle(hues)
    color_map = {}
    for i, key in enumerate(keys):
        h = hues[i]
        s = random.randint(200, 255)
        v = random.randint(230, 255)
        color_bgr = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0][0]
        color_map[key] = tuple(int(c) for c in color_bgr)  # 💥 BGR 그대로 저장
    return color_map

def safe_load_json(path):
    try:
        with open(path, "rb") as fb:
            if fb.read(2) == b"\xff\xd8":
                print(f"[⚠️ JPEG 파일 감지 → 건너뜀] {path}")
                return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[⚠️ JSON 로드 실패: {path}] {e}")
        return None

def polygon_area(points):
    pts = np.array(points, np.float32)
    if len(pts) < 3:
        return 0.0
    return abs(cv2.contourArea(pts))

# ====== 시각화 함수 ======
def visualize_segmentation(image_path, json_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[⚠️ 이미지 읽기 실패] {image_path}")
        return

    data = safe_load_json(json_path)
    if not data or "shapes" not in data:
        return

    h, w = img.shape[:2]

    # ===== 색상 키 생성 =====
    color_keys = []
    for shape in data["shapes"]:
        label = safe_str(shape.get("label")).lower()
        if not label or label == "background":
            continue
        gid = safe_str(shape.get("group_id")) or "no_gid"
        desc_raw = safe_str(shape.get("description")).lower()
        if "ishole" in desc_raw or "is_hole" in desc_raw:
            desc = "ishole"
        elif "iscrowd" in desc_raw or "is_crowd" in desc_raw:
            desc = "iscrowd"
        else:
            desc = ""
        key = f"{label}:{gid}_{desc or 'none'}"
        color_keys.append(key)
    color_map = generate_distinct_colors(sorted(set(color_keys)))

    # ===== legend 구성 =====
    legend_labels = {}

    # ===== 면적 및 우선순위 기반 정렬 =====
    def sort_key(shape):
        label = safe_str(shape.get("label")).lower()
        area = polygon_area(shape.get("points", []))
        priority = 0 if label in ["refrigerator", "background", "wall"] else 1
        area_key = -area if priority == 1 else area
        return (priority, area_key)

    shapes_sorted = sorted(data["shapes"], key=sort_key)

    # ===== RGBA overlay 초기화 =====
    overlay_rgba = np.zeros((h, w, 4), np.uint8)
    alpha_val = 0.55

    for shape in shapes_sorted:
        label = safe_str(shape.get("label"))
        if not label or label.lower() == "background":
            continue

        gid = safe_str(shape.get("group_id")) or "no_gid"
        desc_raw = safe_str(shape.get("description")).lower()
        if "ishole" in desc_raw or "is_hole" in desc_raw:
            desc = "ishole"
        elif "iscrowd" in desc_raw or "is_crowd" in desc_raw:
            desc = "iscrowd"
        else:
            desc = ""

        key = f"{label}:{gid}_{desc or 'none'}"
        color_bgr = color_map.get(key, (0, 255, 0))  # 💥 BGR 유지

        legend_label = f"{label}"
        if gid != "no_gid":
            legend_label += f":{gid}"
        if desc:
            legend_label += f" ({desc})"
        legend_labels[legend_label] = color_bgr  # 💥 legend도 BGR 그대로 저장

        pts = np.array(
            [[max(0, min(w - 1, x)), max(0, min(h - 1, y))]
             for x, y in shape["points"] if math.isfinite(x) and math.isfinite(y)],
            np.int32,
        )
        if len(pts) < 3:
            continue
        pts = pts.reshape((-1, 1, 2))

        # ---- RGBA 마스크 생성 ----
        mask = np.zeros((h, w, 4), np.uint8)
        b, g, r = color_bgr  # 💥 순서 그대로 유지
        cv2.fillPoly(mask, [pts], (b, g, r, int(255 * alpha_val)))
        cv2.polylines(mask, [pts], True, (0, 0, 0, 255), 1)

        # ---- 알파 블렌딩 ----
        alpha = mask[:, :, 3:] / 255.0
        overlay_rgba[:, :, :3] = (1 - alpha) * overlay_rgba[:, :, :3] + alpha * mask[:, :, :3]
        overlay_rgba[:, :, 3:] = np.clip(overlay_rgba[:, :, 3:] + alpha * 255, 0, 255)

    # ===== 원본과 합성 =====
    base = img.astype(float)
    overlay_rgb = overlay_rgba[:, :, :3].astype(float)
    alpha_overlay = np.clip(overlay_rgba[:, :, 3:] / 255.0, 0, 1)
    blended = (1 - alpha_overlay) * base + alpha_overlay * overlay_rgb
    vis = blended.astype(np.uint8)

    # ===== legend =====
    font_scale = 0.6
    thickness = 2
    color_box_w = 25
    padding = 20
    row_gap = 35
    col_gap = 25
    max_text_height = cv2.getTextSize("A", cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][1]
    legend_h = 150
    legend_w = w
    legend_canvas = np.ones((legend_h, legend_w, 3), np.uint8) * 255

    x_cursor = padding
    y_cursor = padding + max_text_height + 10
    max_y_reached = y_cursor

    for lbl, color_bgr in sorted(legend_labels.items()):
        (text_w, text_h), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        block_w = color_box_w + col_gap + text_w + col_gap
        if x_cursor + block_w > legend_w - padding:
            x_cursor = padding
            y_cursor += row_gap
            max_y_reached = y_cursor

        # 💥 legend도 BGR 그대로 사용 (색상 반전 제거)
        cv2.rectangle(legend_canvas, (x_cursor, y_cursor - text_h),
                      (x_cursor + color_box_w, y_cursor + 5), color_bgr, -1)
        cv2.rectangle(legend_canvas, (x_cursor, y_cursor - text_h),
                      (x_cursor + color_box_w, y_cursor + 5), (0, 0, 0), 1)
        cv2.putText(legend_canvas, lbl,
                    (x_cursor + color_box_w + col_gap, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        x_cursor += block_w

        if y_cursor + row_gap > legend_canvas.shape[0] - padding:
            extra = np.ones((100, legend_w, 3), np.uint8) * 255
            legend_canvas = np.vstack([legend_canvas, extra])

    legend_canvas = legend_canvas[:max_y_reached + row_gap, :]

    # ===== 결과 저장 =====
    combined = np.vstack([vis, legend_canvas])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, combined)
    print(f"[✅ 시각화 완료: 색상 일치 + refrigerator 뒤 + 큰 면적 우선] {save_path}")

# ====== 일괄 시각화 ======
for fname in os.listdir(root_dir):
    if not fname.lower().endswith(".jpeg"):
        continue
    base = os.path.splitext(fname)[0]
    json_path = os.path.join(root_dir, base + ".json")
    image_path = os.path.join(root_dir, fname)
    if not os.path.exists(json_path):
        print(f"[⏭️ JSON 없음] {fname}")
        continue
    save_path = os.path.join(save_root, f"{base}_vis.jpeg")
    visualize_segmentation(image_path, json_path, save_path)
