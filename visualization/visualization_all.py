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
 

# ====== 유틸 함수 ======
def safe_str(v):
    """None 또는 비문자열을 안전하게 문자열로 변환"""
    if isinstance(v, str):
        return v.strip()
    if v is None:
        return ""
    return str(v).strip()


def generate_distinct_colors(keys):
    """입력 key 리스트에 대해 색상 대비가 큰 BGR 컬러맵 생성"""
    n = len(keys)
    if n == 0:
        return {}
    # Hue를 균등하게 나누어 높은 대비 확보
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


def safe_load_json(path):
    """JPEG 등 비정상 파일 차단 + 안전한 JSON 로드"""
    try:
        with open(path, "rb") as fb:
            if fb.read(2) == b"\xff\xd8":  # JPEG signature
                print(f"[⚠️ JPEG 파일 감지 → 건너뜀] {path}")
                return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[⚠️ JSON 로드 실패: {path}] {e}")
        return None


# ====== 메인 시각화 함수 ======
# ====== 메인 시각화 함수 ======
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

    # ----------------------------------------------------
    # ① 색상 키 생성 규칙 (변동 없음)
    # ----------------------------------------------------
    color_keys = []
    for shape in data["shapes"]:
        label = safe_str(shape.get("label")).lower()
        if not label or label == "background":
            continue

        gid = shape.get("group_id", None)
        gid_str = safe_str(gid) if gid not in [None, "", "null", "none"] else None
        desc = safe_str(shape.get("description")).lower()

        # ishole/iscrowd 필터
        if not (desc.startswith("ishole") or desc.startswith("iscrowd")):
            desc = ""

        # key는 group_id 존재 여부까지 반영
        key = f"{label}:{gid_str or 'no_gid'}_{desc or 'none'}"
        color_keys.append(key)

    color_keys = sorted(set(color_keys))
    color_map = generate_distinct_colors(color_keys)

    # ----------------------------------------------------
    # ② 폴리곤 그리기 + legend 구성 (변동 없음)
    # ----------------------------------------------------
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

        # legend 이름 구성
        if gid_str:
            legend_label = f"{label}:{gid_str}"
        else:
            legend_label = label
        if desc:
            legend_label += f" ({desc})"

        legend_labels[legend_label] = color

        # polygon 좌표 정리
        pts = np.array(
            [
                [max(0, min(w - 1, x)), max(0, min(h - 1, y))]
                for x, y in shape["points"]
                if math.isfinite(x) and math.isfinite(y)
            ],
            np.int32,
        )
        if len(pts) < 3:
            continue
        pts = pts.reshape((-1, 1, 2))

        # mask 채우기
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(overlay, [pts], True, (0, 0, 0), 1)
        cv2.fillPoly(mask_coverage, [pts], 255)


    # ----------------------------------------------------
    # ③ legend 크기 계산 및 위치 자동 결정 (수정됨)
    # ----------------------------------------------------
    font_scale = 0.6
    legend_gap = 25 # 레이블 간 세로 간격
    padding = 8 # 박스 주변 여백

    # 1. 범례 박스의 최대 너비 계산
    max_text_w = 0
    for lbl in legend_labels.keys():
        (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        max_text_w = max(max_text_w, tw)

    color_box_w = 25 # 색상 박스 너비
    text_start_offset = 35 # 색상 박스와 텍스트 사이 거리
    
    # 총 너비: 왼쪽 여백(8) + 색상 박스(25) + 텍스트 시작 여백(35-25=10) + 최대 텍스트 너비 + 오른쪽 여백(8)
    box_w = int(color_box_w + text_start_offset + max_text_w + padding * 2) 
    box_h = max(40, len(legend_labels) * legend_gap + padding * 2)

    # 2. 범례가 폴리곤을 가리지 않을 최적 위치 탐색 (기존 로직)
    # 이미지를 9등분하여 가장 비어있는 4개의 모서리 중 하나 선택
    corner_regions = {
        "tl": mask_coverage[0:h//3, 0:w//3],
        "tr": mask_coverage[0:h//3, 2*w//3:w],
        "bl": mask_coverage[2*h//3:h, 0:w//3],
        "br": mask_coverage[2*h//3:h, 2*w//3:w],
    }
    free_space = {k: np.sum(v == 0) for k, v in corner_regions.items()}
    best_corner = max(free_space, key=free_space.get)

    # 3. 위치 좌표 결정 (수정됨)
    # 기본 위치: 좌측 15px, 우측 15px, 상단 25px, 하단 20px (기존 설정)
    if best_corner == "tl":
        start_x, start_y = 15, 25
    elif best_corner == "tr":
        start_x, start_y = w - 15 - box_w, 25 # 우상단: box_w를 반영하여 시작점 계산
    elif best_corner == "bl":
        start_x, start_y = 15, h - 20 - box_h + 25 # 하단: box_h를 반영하여 시작점 계산
    else: # br
        start_x, start_y = w - 15 - box_w, h - 20 - box_h + 25 # 우하단: box_w, box_h 반영

    # 4. 프레임 경계 조정 (핵심 수정)
    # X축 조정: 프레임을 벗어나지 않도록 min/max 적용
    legend_x = max(padding, min(w - box_w - padding, start_x))
    # Y축 조정: 프레임을 벗어나지 않도록 min/max 적용
    legend_y = max(padding + 25, min(h - box_h - padding + 25, start_y))


    # ----------------------------------------------------
    # ④ legend 배경 및 텍스트 (수정됨)
    # ----------------------------------------------------
    
    # 텍스트가 시작되는 Y 좌표 (첫 줄의 베이스라인)
    legend_start_y = legend_y 
    # 박스의 좌상단 (x1, y1)
    box_x1 = legend_x - padding
    box_y1 = legend_y - 25 - padding 
    # 박스의 우하단 (x2, y2)
    box_x2 = legend_x - padding + box_w
    box_y2 = legend_y - 25 + box_h - padding
    
    legend_bg = overlay.copy()
    cv2.rectangle(
        legend_bg,
        (box_x1, box_y1),
        (box_x2, box_y2),
        (255, 255, 255),
        -1,
    )
    # 배경에 투명도 적용
    overlay = cv2.addWeighted(overlay, 1.0, legend_bg, 0.9, 0)

    # 범례 항목 그리기
    for i, (lbl, color) in enumerate(sorted(legend_labels.items())):
        # 텍스트 베이스라인 Y 좌표
        y_offset = legend_start_y + i * legend_gap
        
        # 색상 박스 좌표
        color_box_x1 = legend_x
        color_box_y1 = y_offset - 15
        color_box_x2 = legend_x + color_box_w
        color_box_y2 = y_offset + 5
        
        cv2.rectangle(
            overlay,
            (color_box_x1, color_box_y1),
            (color_box_x2, color_box_y2),
            color,
            -1,
        )
        cv2.rectangle(
            overlay,
            (color_box_x1, color_box_y1),
            (color_box_x2, color_box_y2),
            (0, 0, 0),
            1,
        )
        
        # 텍스트 그리기
        cv2.putText(
            overlay,
            lbl,
            (legend_x + text_start_offset, y_offset + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    # ----------------------------------------------------
    # ⑤ 이미지 합성 및 저장 (변동 없음)
    # ----------------------------------------------------
    vis = cv2.addWeighted(img, 0.45, overlay, 0.55, 0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, vis)
    print(f"[✅ 저장 완료] {save_path}")


# ====== 전체 이미지 일괄 시각화 (변동 없음) ======
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