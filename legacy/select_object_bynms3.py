# annotation_nms.py

import os
import json
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
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
    base_annotator: str,
    available_annotators: List[str]
) -> Tuple[List[Dict], str, Dict[str, float], Dict[str, any]]:
    """
    하나의 object에 대한 어노테이션 후보 중 최적의 것을 선택합니다.
    평균 IoU가 가장 높은 후보를 선택합니다.
    
    Parameters:
        candidates: [(mask, shapes, annotator_name), ...] 리스트
        base_annotator: 기준 어노테이터 이름
        available_annotators: 이미지에서 실제 존재하는 모든 어노테이터 리스트
    
    Returns:
        (선택된 shapes, 선택된 annotator_name, pairwise IoU 딕셔너리, 매칭 상태 정보)
    """
    n = len(candidates)
    n_available = len(available_annotators)
    
    # 매칭 상태 정보
    matching_info = {
        'num_candidates': n,
        'num_available_annotators': n_available,
        'is_base_only': n == 1,
        'matched_annotators': [c[2] for c in candidates],
        'matching_status': 'full_match' if n == n_available else 'partial_match' if n >= 2 else 'base_only'
    }
    
    if n == 0:
        return [], "", {}, matching_info
    
    if n == 1:
        # base 어노테이터만 있는 경우
        return candidates[0][1], candidates[0][2], {}, matching_info
    
    # Pairwise IoU 계산 (실제 존재하는 후보들 간에만)
    pairwise_ious = {}
    annotator_names = [c[2] for c in candidates]
    
    for i in range(n):
        for j in range(i + 1, n):
            iou = compute_iou(candidates[i][0], candidates[j][0])
            key = f"{annotator_names[i]}-{annotator_names[j]}"
            pairwise_ious[key] = iou
    
    # 각 후보와 나머지 후보들 간의 평균 IoU 계산
    avg_ious = []
    for i in range(n):
        mask_i = candidates[i][0]
        ious = []
        for j in range(n):
            if i != j:
                iou = compute_iou(mask_i, candidates[j][0])
                ious.append(iou)
        avg_ious.append(np.mean(ious) if ious else 0.0)
    
    # 평균 IoU가 가장 높은 후보 선택
    best_idx = np.argmax(avg_ious)
    matching_info['selection_reason'] = f"avg_iou={avg_ious[best_idx]:.3f}"
    
    return candidates[best_idx][1], candidates[best_idx][2], pairwise_ious, matching_info


def match_objects_across_annotators(
    annotations: List[Dict],
    height: int,
    width: int
) -> Tuple[List[List[Tuple[np.ndarray, List[Dict], str]]], str, Dict[str, int], List[str]]:
    """
    어노테이터들의 object를 Hungarian algorithm으로 최적 매칭합니다.
    가장 많은 object를 표시한 어노테이터를 기준으로 합니다.
    
    Returns:
        (각 object별 후보 리스트, base 어노테이터 이름, 통계 정보, 사용 가능한 어노테이터 리스트)
    """
    if not annotations:
        return [], "", {}, []
    
    # 각 어노테이터별로 object 그룹 생성
    annotator_objects = []
    annotator_names = []
    
    for anno in annotations:
        groups = merge_polygons_by_group(anno['shapes'])
        objects = []
        for gid, shapes in groups.items():
            mask = create_object_mask(shapes, height, width)
            objects.append((mask, shapes, anno['annotator']))
        annotator_objects.append(objects)
        annotator_names.append(anno['annotator'])
    
    # 통계 정보
    num_objects = [len(objs) for objs in annotator_objects]
    stats = {
        'annotator_object_counts': dict(zip(annotator_names, num_objects)),
        'total_unique_objects': max(num_objects) if num_objects else 0,
        'num_annotators': len(annotator_names)
    }
    
    if len(set(num_objects)) != 1:
        print(f"경고: 어노테이터별 object 수가 다릅니다: {dict(zip(annotator_names, num_objects))}")
    
    # 가장 많은 object를 표시한 어노테이터를 base로 선택
    base_idx = np.argmax(num_objects)
    base_annotator = annotator_names[base_idx]
    base_objects = annotator_objects[base_idx]
    n_base = len(base_objects)
    
    print(f"Base 어노테이터: {base_annotator} ({n_base}개 object)")
    print(f"사용 가능한 어노테이터: {', '.join(annotator_names)} (총 {len(annotator_names)}명)")
    
    # 단일 어노테이터인 경우
    if len(annotator_names) == 1:
        matched = [[(obj[0], obj[1], obj[2])] for obj in base_objects]
        return matched, base_annotator, stats, annotator_names
    
    # 다른 어노테이터들의 인덱스
    other_indices = [i for i in range(len(annotations)) if i != base_idx]
    
    matched = []
    
    for base_idx_obj in range(n_base):
        base_mask, base_shapes, base_name = base_objects[base_idx_obj]
        candidates = [(base_mask, base_shapes, base_name)]
        
        # 각 다른 어노테이터와 Hungarian matching 수행
        for other_idx in other_indices:
            other_objects = annotator_objects[other_idx]
            n_other = len(other_objects)
            
            if n_other == 0:
                print(f"  Object {base_idx_obj}: {annotator_names[other_idx]}는 object가 없습니다.")
                continue
            
            # IoU cost matrix 생성 (maximize IoU = minimize -IoU)
            cost_matrix = np.zeros((n_base, n_other))
            for i in range(n_base):
                for j in range(n_other):
                    iou = compute_iou(base_objects[i][0], other_objects[j][0])
                    cost_matrix[i, j] = -iou  # negative for minimization
            
            # Hungarian algorithm으로 최적 매칭
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # 현재 base_idx_obj에 매칭된 object 찾기
            if base_idx_obj < len(col_ind):
                matched_idx = col_ind[base_idx_obj]
                matched_iou = -cost_matrix[base_idx_obj, matched_idx]
                
                if matched_iou > 0.1:  # minimum IoU threshold
                    candidates.append(other_objects[matched_idx])
                else:
                    print(f"  Object {base_idx_obj}: {annotator_names[other_idx]}와 매칭 실패 (IoU={matched_iou:.3f})")
            else:
                print(f"  Object {base_idx_obj}: {annotator_names[other_idx]}에 대응되는 object 없음")
        
        matched.append(candidates)
    
    return matched, base_annotator, stats, annotator_names


def process_image_annotations(
    annotator_folders: List[str],
    image_name: str,
    output_folder: str,
    iou_records: List[Dict],
    exception_records: List[Dict]
) -> None:
    """
    하나의 이미지에 있는 각 object별로 어노테이션을 통합합니다.
    가장 많은 object를 표시한 어노테이터를 기준으로 합니다.
    존재하는 JSON 파일만 사용하여 처리합니다.
    """
    # 존재하는 JSON 파일만 로드
    annotations = []
    available_folders = []
    missing_folders = []
    
    for folder in annotator_folders:
        json_path = os.path.join(folder, image_name)
        if os.path.exists(json_path):
            try:
                anno = load_json(json_path)
                annotations.append(anno)
                available_folders.append(os.path.basename(folder))
            except Exception as e:
                print(f"경고: {json_path} 로드 실패 - {e}")
                missing_folders.append(os.path.basename(folder))
        else:
            missing_folders.append(os.path.basename(folder))
    
    # 최소 1개 이상의 어노테이션이 필요
    if not annotations:
        print(f"경고: {image_name}에 대한 어노테이션이 하나도 없습니다. 건너뜁니다.")
        return
    
    print(f"사용 가능: {', '.join(available_folders)}")
    if missing_folders:
        print(f"누락됨: {', '.join(missing_folders)}")
    
    # 이미지 크기 확인
    height = annotations[0]['imageHeight']
    width = annotations[0]['imageWidth']
    
    # Object 매칭 및 NMS 수행
    matched_objects, base_annotator, stats, available_annotators = \
        match_objects_across_annotators(annotations, height, width)
    
    # 최종 결과 생성 및 기록
    final_shapes = []
    full_matches = 0
    partial_matches = 0
    base_only = 0
    
    n_available = len(available_annotators)
    
    for obj_idx, candidates in enumerate(matched_objects):
        selected_shapes, selected_annotator, pairwise_ious, matching_info = \
            select_best_annotation(candidates, base_annotator, available_annotators)
        
        final_shapes.extend(selected_shapes)
        
        # 매칭 상태 카운트
        if matching_info['matching_status'] == 'full_match':
            full_matches += 1
        elif matching_info['matching_status'] == 'partial_match':
            partial_matches += 1
        else:
            base_only += 1
        
        # IoU 기록 저장
        record = {
            'image_name': image_name,
            'object_id': obj_idx,
            'num_candidates': len(candidates),
            'num_available_annotators': n_available,
            'selected_annotator': selected_annotator,
            'base_annotator': base_annotator,
            'matching_status': matching_info['matching_status'],
            'matched_annotators': ','.join(matching_info['matched_annotators']),
            'available_annotators': ','.join(available_annotators),
            'missing_annotators': ','.join(missing_folders) if missing_folders else 'none',
        }
        
        # Pairwise IoU 추가 (실제 존재하는 페어만)
        record.update(pairwise_ious)
        
        # 사용 가능한 어노테이터들 간의 모든 페어 생성
        all_pairs = [
            f"{available_annotators[i]}-{available_annotators[j]}"
            for i in range(len(available_annotators))
            for j in range(i + 1, len(available_annotators))
        ]
        
        # 계산되지 않은 페어는 N/A로 표시
        for pair in all_pairs:
            if pair not in record:
                record[pair] = 'N/A'
        
        iou_records.append(record)
        
        # 예외 상황 기록 (완전 매칭이 아닌 경우)
        if matching_info['matching_status'] != 'full_match':
            missing_in_match = [a for a in available_annotators 
                               if a not in matching_info['matched_annotators']]
            
            exception_record = {
                'image_name': image_name,
                'object_id': obj_idx,
                'base_annotator': base_annotator,
                'matching_status': matching_info['matching_status'],
                'num_matched': len(candidates),
                'num_available': n_available,
                'matched_annotators': ','.join(matching_info['matched_annotators']),
                'missing_in_match': ','.join(missing_in_match) if missing_in_match else 'none',
                'missing_files': ','.join(missing_folders) if missing_folders else 'none',
                'selected_annotator': selected_annotator,
                'reason': f"Incomplete consensus ({len(candidates)}/{n_available} matched)"
            }
            exception_records.append(exception_record)
    
    # 이미지별 통계 추가
    image_stats = {
        'image_name': image_name,
        'base_annotator': base_annotator,
        'num_available_annotators': n_available,
        'available_annotators': ','.join(available_annotators),
        'missing_files': ','.join(missing_folders) if missing_folders else 'none',
        'total_objects': len(matched_objects),
        'full_matches': full_matches,
        'partial_matches': partial_matches,
        'base_only': base_only,
    }
    image_stats.update(stats['annotator_object_counts'])
    exception_records.append(image_stats)
    
    # 결과 JSON 생성
    result = {
        "version": annotations[0]['version'],
        "flags": annotations[0]['flags'],
        "shapes": final_shapes,
        "imagePath": annotations[0]['imagePath'],
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "annotator": "consensus",
        "base_annotator": base_annotator,
        "available_annotators": available_annotators,
        "missing_annotators": missing_folders,
        "matching_statistics": {
            "num_available_annotators": n_available,
            "total_objects": len(matched_objects),
            "full_matches": full_matches,
            "partial_matches": partial_matches,
            "base_only": base_only
        }
    }
    
    # 저장
    output_path = os.path.join(output_folder, image_name)
    os.makedirs(output_folder, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"처리 완료: {image_name}")
    print(f"  총 {len(final_shapes)}개 폴리곤")
    print(f"  완전 매칭: {full_matches}, 부분 매칭: {partial_matches}, Base만: {base_only}")


def save_iou_records(iou_records: List[Dict], output_path: str) -> None:
    """
    IoU 기록을 CSV 파일로 저장합니다.
    """
    if not iou_records:
        print("경고: 저장할 IoU 기록이 없습니다.")
        return
    
    # 모든 가능한 키 수집
    all_keys = set()
    for record in iou_records:
        all_keys.update(record.keys())
    
    # 고정 컬럼 순서
    fixed_cols = ['image_name', 'object_id', 'num_candidates', 'num_available_annotators',
                  'selected_annotator', 'base_annotator', 'matching_status', 
                  'matched_annotators', 'available_annotators', 'missing_annotators']
    iou_cols = sorted([k for k in all_keys if '-' in k and k not in fixed_cols])
    other_cols = sorted([k for k in all_keys if k not in fixed_cols and k not in iou_cols])
    
    fieldnames = fixed_cols + iou_cols + other_cols
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(iou_records)
    
    print(f"\nIoU 기록이 저장되었습니다: {output_path}")


def save_exception_records(exception_records: List[Dict], output_path: str) -> None:
    """
    예외 상황 기록을 CSV 파일로 저장합니다.
    """
    if not exception_records:
        print("경고: 저장할 예외 기록이 없습니다.")
        return
    
    # 통계 레코드와 예외 레코드 분리
    stats_records = [r for r in exception_records if 'total_objects' in r]
    exception_only = [r for r in exception_records if 'total_objects' not in r]
    
    # 예외 레코드 저장
    if exception_only:
        exception_fieldnames = ['image_name', 'object_id', 'base_annotator', 
                               'matching_status', 'num_matched', 'num_available',
                               'matched_annotators', 'missing_in_match', 'missing_files',
                               'selected_annotator', 'reason']
        
        exception_path = output_path.replace('.csv', '_exceptions.csv')
        with open(exception_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=exception_fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(exception_only)
        
        print(f"예외 상황 기록이 저장되었습니다: {exception_path}")
    
    # 통계 레코드 저장
    if stats_records:
        all_keys = set()
        for record in stats_records:
            all_keys.update(record.keys())
        
        stats_fieldnames = ['image_name', 'base_annotator', 'num_available_annotators',
                           'available_annotators', 'missing_files',
                           'total_objects', 'full_matches', 'partial_matches', 'base_only']
        annotator_cols = sorted([k for k in all_keys if k not in stats_fieldnames])
        stats_fieldnames.extend(annotator_cols)
        
        stats_path = output_path.replace('.csv', '_statistics.csv')
        with open(stats_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=stats_fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(stats_records)
        
        print(f"이미지별 통계가 저장되었습니다: {stats_path}")


def main(root_folder: str, output_folder: str, iou_output_path: str) -> None:
    """
    메인 처리 함수입니다.
    
    Parameters:
        root_folder: annotator1, annotator2, annotator3 폴더가 있는 루트 폴더
        output_folder: 통합 결과를 저장할 폴더
        iou_output_path: IoU 기록을 저장할 CSV 파일 경로
    """
    annotator_folders = [
        os.path.join(root_folder, "annotator1"),
        os.path.join(root_folder, "annotator2"),
        os.path.join(root_folder, "annotator3")
    ]
    
    # 모든 어노테이터 폴더에서 JSON 파일 수집
    all_json_files = set()
    for folder in annotator_folders:
        if os.path.exists(folder):
            json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
            all_json_files.update(json_files)
    
    if not all_json_files:
        print(f"오류: 어노테이터 폴더에 JSON 파일이 없습니다.")
        return
    
    all_json_files = sorted(all_json_files)
    
    print(f"총 {len(all_json_files)}개 이미지를 처리합니다.\n")
    
    iou_records = []
    exception_records = []
    
    for json_file in all_json_files:
        print(f"\n{'='*60}")
        print(f"처리 중: {json_file}")
        print(f"{'='*60}")
        process_image_annotations(
            annotator_folders, json_file, output_folder, 
            iou_records, exception_records
        )
    
    # IoU 기록 저장
    save_iou_records(iou_records, iou_output_path)
    
    # 예외 및 통계 기록 저장
    save_exception_records(exception_records, iou_output_path)
    
    print(f"\n{'='*60}")
    print(f"모든 처리가 완료되었습니다.")
    print(f"{'='*60}")
    print(f"통합 어노테이션: {output_folder}")
    print(f"IoU 기록: {iou_output_path}")
    print(f"예외 기록: {iou_output_path.replace('.csv', '_exceptions.csv')}")
    print(f"통계 기록: {iou_output_path.replace('.csv', '_statistics.csv')}")


if __name__ == "__main__":
    root = "/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st_phase2_validation"
    output = "/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st_phase2_validation_feedback"
    iou_csv = "/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st_phase2_validation_feedback/iou_records.csv"
    main(root, output, iou_csv)