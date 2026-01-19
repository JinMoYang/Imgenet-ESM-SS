import os
import json
import cv2
import numpy as np
import random
import math

# ====== 경로 설정 ======
root_dir = "/ssd/dywoo/IM-meta/phase2/batch_1st_phase2_validation"
annotator_folders = ["annotator1", "annotator2", "annotator3"]
merged_folder = os.path.join(root_dir, "merged")

# ====== 유틸 ======
def safe_str(v):
    if isinstance(v, str):
        return v.strip()
    if v is None:
        return ""
    return str(v).strip()

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

def generate_distinct_colors(keys):
    n = len(keys)
    if n == 0:
        return {}
    hues = np.linspace(0, 179, n, endpoint=False)
    random.shuffle(hues)
    color_map = {}
    for i, key in enumerate(keys):
        h = hues[i]
        s = random.randint(200, 255)
        v = random.randint(230, 255)
        color = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0][0]
        color_map[key] = tuple(int(c) for c in color)
    return color_map


# ====== 시각화 함수 ======
def visualize_segmentation(image_path, json_path, save_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"[⚠️ 이미지 읽기 실패] {image_path}")
        return

    data = safe_load_json(json_path)
    if not data or "shapes" not in data:
        return

    overlay = img.copy()
    h, w = img.shape[:2]
    mask_coverage = np.zeros((h, w), np.uint8)

    # 색상 키 생성
    color_keys = []
    for shape in data["shapes"]:
        label = safe_str(shape.get("label")).lower()
        if not label or label == "background":
            continue
        gid = shape.get("group_id", None)
        gid_str = safe_str(gid) if gid not in [None, "", "null", "none"] else None
        desc = safe_str(shape.get("description")).lower()
        if not (desc.startswith("ishole") or desc.startswith("iscrowd")):
            desc = ""
        key = f"{label}:{gid_str or 'no_gid'}_{desc or 'none'}"
        color_keys.append(key)

    color_map = generate_distinct_colors(sorted(set(color_keys)))
    legend_labels = {}

    for shape in data["shapes"]:
        label = safe_str(shape.get("label"))
        if not label or label.lower() == "background":
            continue

        gid = shape.get("group_id", None)
        gid_str = safe_str(gid) if gid not in [None, "", "null", "none"] else None
        desc = safe_str(shape.get("description")).lower()
        if not (desc.startswith("ishole") or desc.startswith("iscrowd")):
            desc = ""

        key = f"{label}:{gid_str or 'no_gid'}_{desc or 'none'}"
        color = color_map.get(key, (0, 255, 0))

        legend_label = f"{label}:{gid_str}" if gid_str else label
        if desc:
            legend_label += f" ({desc})"
        legend_labels[legend_label] = color

        pts = np.array(
            [[max(0, min(w - 1, x)), max(0, min(h - 1, y))]
             for x, y in shape["points"] if math.isfinite(x) and math.isfinite(y)],
            np.int32,
        )
        if len(pts) < 3:
            continue
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(overlay, [pts], True, (0, 0, 0), 1)
        cv2.fillPoly(mask_coverage, [pts], 255)

    # legend 위치 결정
    corner_regions = {
        "tl": mask_coverage[0:h//3, 0:w//3],
        "tr": mask_coverage[0:h//3, 2*w//3:w],
        "bl": mask_coverage[2*h//3:h, 0:w//3],
        "br": mask_coverage[2*h//3:h, 2*w//3:w],
    }
    free_space = {k: np.sum(v == 0) for k, v in corner_regions.items()}
    best_corner = max(free_space, key=free_space.get)
    if best_corner == "tl":
        legend_x, legend_y = 15, 25
    elif best_corner == "tr":
        legend_x, legend_y = w - 160, 25
    elif best_corner == "bl":
        legend_x, legend_y = 15, h - 20 - (len(legend_labels) * 25)
    else:
        legend_x, legend_y = w - 160, h - 20 - (len(legend_labels) * 25)

    # legend 박스
    font_scale = 0.6
    max_text_w = 0
    for lbl in legend_labels.keys():
        (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        max_text_w = max(max_text_w, tw)
    box_w = int(40 + max_text_w)
    box_h = max(40, len(legend_labels) * 25 + 20)
    cv2.rectangle(
        overlay,
        (legend_x - 8, legend_y - 25),
        (legend_x + box_w, legend_y + box_h - 20),
        (255, 255, 255),
        -1,
    )

    for i, (lbl, color) in enumerate(sorted(legend_labels.items())):
        y_offset = legend_y + i * 25
        cv2.rectangle(overlay, (legend_x, y_offset - 15), (legend_x + 25, y_offset + 5), color, -1)
        cv2.rectangle(overlay, (legend_x, y_offset - 15), (legend_x + 25, y_offset + 5), (0, 0, 0), 1)
        cv2.putText(
            overlay,
            lbl,
            (legend_x + 35, y_offset + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    vis = cv2.addWeighted(img, 0.45, overlay, 0.55, 0)
    cv2.imwrite(save_path, vis)
    print(f"[✅ 저장 완료] {save_path}")


# ====== ① annotator1~3 ======
for annotator in annotator_folders:
    annotator_path = os.path.join(root_dir, annotator)
    if not os.path.isdir(annotator_path):
        continue

    for fname in os.listdir(annotator_path):
        if not fname.lower().endswith((".jpeg", ".jpg", ".png")):
            continue

        img_path = os.path.join(annotator_path, fname)
        json_path = os.path.join(annotator_path, os.path.splitext(fname)[0] + ".json")
        if not os.path.exists(json_path):
            print(f"[⏭️ JSON 없음] {fname}")
            continue

        data = safe_load_json(json_path)
        if not data:
            continue

        annotator_name = data.get("annotator", annotator)
        base, ext = os.path.splitext(fname)
        save_fname = f"{base}_{annotator_name}.jpeg"
        save_path = os.path.join(annotator_path, save_fname)  # ✅ 같은 폴더에 저장
        visualize_segmentation(img_path, json_path, save_path)


# ====== ② merged 폴더 ======
if os.path.isdir(merged_folder):
    for json_file in os.listdir(merged_folder):
        if not json_file.lower().endswith(".json"):
            continue

        json_path = os.path.join(merged_folder, json_file)
        base_name = os.path.splitext(json_file)[0] + ".JPEG"

        # 이미지 찾기
        found_image = None
        for annotator in annotator_folders:
            candidate = os.path.join(root_dir, annotator, base_name)
            if os.path.exists(candidate):
                found_image = candidate
                break

        if not found_image:
            print(f"[⚠️ 매칭 이미지 없음] {json_file}")
            continue

        data = safe_load_json(json_path)
        if not data:
            continue

        all_annotators = set()
        for shape in data.get("shapes", []):
            for ann in shape.get("annotators", []):
                if ann:
                    all_annotators.add(ann)
        top_annotator = data.get("annotator")
        if top_annotator and top_annotator != "merged":
            all_annotators.add(top_annotator)

        annotators_list = sorted(list(all_annotators))
        annotator_tag = "_".join(annotators_list) if annotators_list else "merged"

        save_fname = f"{os.path.splitext(base_name)[0]}_{annotator_tag}.jpeg"
        save_path = os.path.join(merged_folder, save_fname)  # ✅ merged 폴더 안에 저장
        visualize_segmentation(found_image, json_path, save_path)
