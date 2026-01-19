# annotation_nms.py

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment


def load_json(path: str) -> Dict:
    """
    JSON 파일을 읽어서 딕셔너리로 반환합니다.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"파일을 찾을 수 없습니다: {path}")
    except json.JSONDecodeError:
        raise ValueError(f"JSON 파싱 오류: {path}")


def polygon_to_mask(points: List[List[float]], height: int, width: int) -> np.ndarray:
    """
    폴리곤 좌표를 binary mask로 변환합니다.
    
    Parameters:
        points: [[x1, y1], [x2, y2], ...] 형태의 폴리곤 좌표
        height: 이미지 높이
        width: 이미지 너비
    
    Returns:
        (height, width) shape의 binary mask
    """
    mask = Image.new('L', (width, height), 0)
    flat_points = [(x, y) for x, y in points]
    ImageDraw.Draw(mask).polygon(flat_points, outline=1, fill=1)
    return np.array(mask, dtype=bool)


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    두 binary mask 간의 IoU를 계산합니다.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union


def merge_polygons_by_group(shapes: List[Dict]) -> Dict[int, List[Dict]]:
    """
    shapes 리스트를 group_id별로 묶습니다.
    group_id가 None인 경우 각 shape를 별도의 그룹으로 처리합니다.
    
    Returns:
        {group_id: [shape1, shape2, ...]} 형태의 딕셔너리
    """
    groups = defaultdict(list)
    next_auto_id = 0
    
    for shape in shapes:
        gid = shape.get('group_id')
        if gid is None:
            gid = f"auto_{next_auto_id}"
            next_auto_id += 1
        groups[gid].append(shape)
    
    return groups


def create_object_mask(group_shapes: List[Dict], height: int, width: int) -> np.ndarray:
    """
    같은 group_id를 가진 여러 폴리곤을 하나의 mask로 통합합니다.
    """
    mask = np.zeros((height, width), dtype=bool)
    for shape in group_shapes:
        poly_mask = polygon_to_mask(shape['points'], height, width)
        mask = np.logical_or(mask, poly_mask)
    return mask


def select_best_annotation(
    candidates: List[Tuple[np.ndarray, List[Dict], str]],
) -> Tuple[List[Dict], str]:
    """
    하나의 object에 대한 3개 어노테이션 후보 중 최적의 것을 선택합니다.
    평균 IoU가 가장 높은 후보를 선택합니다.
    
    Parameters:
        candidates: [(mask, shapes, annotator_name), ...] 리스트
    
    Returns:
        (선택된 shapes, 선택된 annotator_name)
    """
    n = len(candidates)
    if n == 0:
        return [], ""
    if n == 1:
        return candidates[0][1], candidates[0][2]
    
    # 각 후보와 나머지 후보들 간의 평균 IoU 계산
    avg_ious = []
    for i in range(n):
        mask_i = candidates[i][0]
        ious = []
        for j in range(n):
            if i != j:
                iou = compute_iou(mask_i, candidates[j][0])
                ious.append(iou)
        avg_ious.append(np.mean(ious))
    
    # 평균 IoU가 가장 높은 후보 선택
    best_idx = np.argmax(avg_ious)
    return candidates[best_idx][1], candidates[best_idx][2]


def match_objects_across_annotators(
    annotations: List[Dict],
    height: int,
    width: int
) -> List[List[Tuple[np.ndarray, List[Dict], str]]]:
    """
    세 어노테이터의 object들을 Hungarian algorithm으로 최적 매칭합니다.
    각 object별로 3개 어노테이션을 하나씩만 매칭합니다.
    
    Returns:
        각 object별로 [(mask, shapes, annotator), ...] 리스트를 담은 리스트
    """
    # 각 어노테이터별로 object 그룹 생성
    annotator_objects = []
    for anno in annotations:
        groups = merge_polygons_by_group(anno['shapes'])
        objects = []
        for gid, shapes in groups.items():
            mask = create_object_mask(shapes, height, width)
            objects.append((mask, shapes, anno['annotator']))
        annotator_objects.append(objects)
    
    # 모든 어노테이터의 object 수가 같은지 확인
    num_objects = [len(objs) for objs in annotator_objects]
    if len(set(num_objects)) != 1:
        print(f"경고: 어노테이터별 object 수가 다릅니다: {num_objects}")
    
    # 첫 번째 어노테이터를 기준으로 매칭
    base_objects = annotator_objects[0]
    n_base = len(base_objects)
    
    matched = []
    
    for base_idx in range(n_base):
        base_mask, base_shapes, base_name = base_objects[base_idx]
        candidates = [(base_mask, base_shapes, base_name)]
        
        # 각 어노테이터별로 Hungarian matching 수행
        for other_objects in annotator_objects[1:]:
            n_other = len(other_objects)
            
            # IoU cost matrix 생성 (maximize IoU = minimize -IoU)
            cost_matrix = np.zeros((n_base, n_other))
            for i in range(n_base):
                for j in range(n_other):
                    iou = compute_iou(base_objects[i][0], other_objects[j][0])
                    cost_matrix[i, j] = -iou  # negative for minimization
            
            # Hungarian algorithm으로 최적 매칭
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # 현재 base_idx에 매칭된 object 찾기
            matched_idx = col_ind[base_idx]
            matched_iou = -cost_matrix[base_idx, matched_idx]
            
            if matched_iou > 0.1:  # minimum IoU threshold
                candidates.append(other_objects[matched_idx])
        
        matched.append(candidates)
    
    return matched


def process_image_annotations(
    annotator_folders: List[str],
    image_name: str,
    output_folder: str
) -> None:
    """
    모든 폴리곤을 전부 비교, IoU가 가장 높은 것들끼리 매칭하여 intersection 부분만 최종 json으로 남김
    하나의 이미지에 있는 각 object별로 3개 어노테이션을 NMS로 통합합니다.
    Hungarian algorithm으로 object를 1:1:1 매칭한 후, 각 object별로 최적 어노테이션을 선택합니다.
    """
    # 각 어노테이터의 json 파일 로드
    annotations = []
    for folder in annotator_folders:
        json_path = os.path.join(folder, image_name)
        if not os.path.exists(json_path):
            print(f"경고: {json_path} 파일이 존재하지 않습니다.")
            return
        anno = load_json(json_path)
        annotations.append(anno)
    
    # 이미지 크기 확인
    height = annotations[0]['imageHeight']
    width = annotations[0]['imageWidth']
    
    # Object 매칭 및 NMS 수행
    matched_objects = match_objects_across_annotators(annotations, height, width)
    
    # 최종 결과 생성
    final_shapes = []
    for candidates in matched_objects:
        selected_shapes, selected_annotator = select_best_annotation(candidates)
        final_shapes.extend(selected_shapes)
    
    # 결과 JSON 생성
    result = {
        "version": annotations[0]['version'],
        "flags": annotations[0]['flags'],
        "shapes": final_shapes,
        "imagePath": annotations[0]['imagePath'],
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "annotator": "consensus"
    }
    
    # 저장
    output_path = os.path.join(output_folder, image_name)
    os.makedirs(output_folder, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"처리 완료: {image_name} (총 {len(final_shapes)}개 폴리곤)")


def main(root_folder: str, output_folder: str) -> None:
    """
    메인 처리 함수입니다.
    
    Parameters:
        root_folder: annotator1, annotator2, annotator3 폴더가 있는 루트 폴더
        output_folder: 통합 결과를 저장할 폴더
    """
    annotator_folders = [
        os.path.join(root_folder, "annotator1"),
        os.path.join(root_folder, "annotator2"),
        os.path.join(root_folder, "annotator3")
    ]
    
    # 첫 번째 어노테이터 폴더의 파일 목록을 기준으로 처리
    base_folder = annotator_folders[0]
    if not os.path.exists(base_folder):
        print(f"오류: {base_folder} 폴더가 존재하지 않습니다.")
        return
    
    json_files = [f for f in os.listdir(base_folder) if f.endswith('.json')]
    
    print(f"총 {len(json_files)}개 이미지를 처리합니다.")
    for json_file in json_files:
        process_image_annotations(annotator_folders, json_file, output_folder)
    
    print(f"\n모든 처리가 완료되었습니다. 결과는 {output_folder}에 저장되었습니다.")


if __name__ == "__main__":
    root = "./annotations"
    output = "./annotations_consensus"
    main(root, output)