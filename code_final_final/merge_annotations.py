# merge_annotations.py
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import csv
from polygon_utils import compute_iou, group_polygons_by_group_id
from key_mapper import KeyMapper
from voting import pixel_wise_voting

def load_annotations(base_dir: str, filename: str) -> List[Dict[str, Any]]:
    """
    여러 annotator 폴더에서 동일한 파일명의 JSON을 로드한다.
    
    Args:
        base_dir: 기본 디렉토리 경로
        filename: JSON 파일명
    
    Returns:
        각 annotator의 JSON 데이터 리스트
    """
    annotations = []
    annotator_dirs = sorted([d for d in Path(base_dir).iterdir() if d.is_dir()])
    
    for idx, ann_dir in enumerate(annotator_dirs):
        json_path = ann_dir / filename
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['annotator_idx'] = idx
                annotations.append(data)
        else:
            print(f"경고: {json_path}를 찾을 수 없습니다.")
    
    return annotations

def match_polygons(annotations: List[Dict[str, Any]], iou_threshold: float = 0.5) -> KeyMapper:
    """
    모든 annotator의 폴리곤을 IoU 기반으로 매칭한다.
    레이블과 무관하게 IoU만으로 매칭을 수행한다.
    
    Args:
        annotations: 각 annotator의 annotation 데이터 리스트
        iou_threshold: IoU threshold
    
    Returns:
        KeyMapper 객체와 그룹화된 annotation
    """
    # 각 annotator의 폴리곤 그룹 생성
    grouped_annotations = []
    for ann_data in annotations:
        groups = group_polygons_by_group_id(ann_data['shapes'])
        grouped_annotations.append({
            'annotator_idx': ann_data['annotator_idx'],
            'groups': groups
        })
    
    # Key mapping 초기화
    key_mapper = KeyMapper()
    
    # 모든 annotator 쌍에 대해 비교
    n = len(grouped_annotations)
    for i in range(n):
        for j in range(i + 1, n):
            ann_i = grouped_annotations[i]
            ann_j = grouped_annotations[j]
            
            # 각 폴리곤 그룹 쌍을 비교
            for idx_i, group_i in enumerate(ann_i['groups']):
                for idx_j, group_j in enumerate(ann_j['groups']):
                    # 레이블 체크 제거 - IoU만으로 매칭
                    iou = compute_iou(group_i['merged_points'], group_j['merged_points'])
                    
                    if iou >= iou_threshold:
                        key_mapper.update(
                            ann_i['annotator_idx'], idx_i,
                            ann_j['annotator_idx'], idx_j
                        )
    
    return key_mapper, grouped_annotations

def create_final_shapes(approved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    승인된 폴리곤 그룹에서 최종 shapes를 생성한다.
    pixel-wise voting으로 생성된 새로운 폴리곤들을 사용한다.
    
    Args:
        approved: 승인된 폴리곤 리스트
    
    Returns:
        최종 shapes 리스트
    """
    final_shapes = []
    next_group_id = 1
    
    for item in approved:
        new_polygons = item['new_polygons']
        label = item['label']
        description = item['description']
        
        if len(new_polygons) == 1:
            # 단일 폴리곤
            shape = {
                'label': label,
                'score': None,
                'points': new_polygons[0],
                'group_id': None,
                'description': description,
                'difficult': False,
                'shape_type': 'polygon',
                'flags': {},
                'attributes': {},
                'kie_linking': []
            }
            final_shapes.append(shape)
        else:
            # 여러 폴리곤 조각으로 구성된 경우 group_id 할당
            current_group_id = next_group_id
            next_group_id += 1
            
            for poly_points in new_polygons:
                shape = {
                    'label': label,
                    'score': None,
                    'points': poly_points,
                    'group_id': current_group_id,
                    'description': description,
                    'difficult': False,
                    'shape_type': 'polygon',
                    'flags': {},
                    'attributes': {},
                    'kie_linking': []
                }
                final_shapes.append(shape)
    
    return final_shapes

def save_voting_results(approved: List[Dict[str, Any]], 
                       rejected: List[Dict[str, Any]], 
                       annotator_names: Dict[int, str],
                       output_path: str):
    """
    투표 결과를 CSV 파일로 저장한다.
    
    Args:
        approved: 승인된 폴리곤 리스트
        rejected: 기각된 폴리곤 리스트
        annotator_names: annotator_idx -> 이름 매핑
        output_path: CSV 출력 경로
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['key', 'label', 'votes', 'status', 'annotators', 'num_polygons'])
        
        for item in approved:
            annotator_list = [annotator_names.get(idx, f'annotator_{idx}') 
                            for idx in item['annotators']]
            writer.writerow([
                item['key'],
                item['label'],
                item['votes'],
                'approved',
                ','.join(annotator_list),
                len(item['new_polygons'])
            ])
        
        for item in rejected:
            annotator_idx = item.get('annotator_idx')
            annotator_name = annotator_names.get(annotator_idx, '') if annotator_idx is not None else ''
            
            writer.writerow([
                item['key'],
                item['data'].get('label', 'unknown'),
                item['votes'],
                'rejected',
                annotator_name,
                0
            ])

def merge_single_file(base_dir: str, filename: str, output_dir: str, 
                     iou_threshold: float = 0.5, min_votes: int = 2):
    """
    단일 JSON 파일에 대해 전체 병합 프로세스를 수행한다.
    
    Args:
        base_dir: annotator 폴더들이 있는 기본 디렉토리
        filename: 병합할 JSON 파일명
        output_dir: 결과 저장 디렉토리
        iou_threshold: IoU threshold
        min_votes: 최소 필요 투표 수
    """
    print(f"처리 중: {filename}")
    
    # 1. JSON 로드
    annotations = load_annotations(base_dir, filename)
    if len(annotations) < 2:
        print(f"경고: {filename}에 대한 annotation이 2개 미만입니다.")
        return
    
    # Annotator 이름 매핑 생성
    annotator_names = {ann['annotator_idx']: ann.get('annotator', f"annotator_{ann['annotator_idx']}") 
                       for ann in annotations}
    
    # 이미지 크기 가져오기
    image_height = annotations[0]['imageHeight']
    image_width = annotations[0]['imageWidth']
    
    # 2. 폴리곤 매칭 (레이블 무관)
    key_mapper, grouped_annotations = match_polygons(annotations, iou_threshold)
    
    # 3. Key별로 그룹화
    groups = key_mapper.get_groups()

    # 3-1. 매칭된 폴리곤 ID 수집
    matched_poly_ids = set()
    for poly_list in groups.values():
        matched_poly_ids.update(poly_list)
    
    # 3-2. 미매칭 폴리곤 찾기
    unmatched_polygons = []
    for ann in grouped_annotations:
        annotator_idx = ann['annotator_idx']
        for poly_idx, group_data in enumerate(ann['groups']):
            if (annotator_idx, poly_idx) not in matched_poly_ids:
                unmatched_polygons.append({
                    'key': 'unmatched',
                    'votes': 1,
                    'data': group_data,
                    'annotator_idx': annotator_idx
                })
    
    # 4. 각 key에 대한 폴리곤 데이터 수집
    polygon_groups = {}
    for key, poly_list in groups.items():
        polygon_groups[key] = []
        for annotator_idx, poly_idx in poly_list:
            # 해당 annotator의 그룹 데이터 찾기
            for ann in grouped_annotations:
                if ann['annotator_idx'] == annotator_idx:
                    poly_data = ann['groups'][poly_idx]
                    polygon_groups[key].append((annotator_idx, poly_idx, poly_data))
                    break
    
    # 5. Pixel-wise majority voting
    approved, rejected = pixel_wise_voting(
        polygon_groups, image_height, image_width, min_votes
    )
    
    # 5-1. 미매칭 폴리곤을 rejected에 추가
    rejected.extend(unmatched_polygons)

    # 6. 최종 shapes 생성
    final_shapes = create_final_shapes(approved)
    
    # 7. 최종 JSON 생성
    base_annotation = annotations[0]
    final_json = {
        'version': base_annotation['version'],
        'flags': base_annotation.get('flags', {}),
        'shapes': final_shapes,
        'imagePath': base_annotation['imagePath'],
        'imageData': None,
        'imageHeight': base_annotation['imageHeight'],
        'imageWidth': base_annotation['imageWidth'],
        'annotator': 'merged'
    }
    
    # 8. 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    output_json_path = os.path.join(output_dir, filename)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    
    # 9. 투표 결과 CSV 저장
    csv_filename = filename.replace('.json', '_voting.csv')
    output_csv_path = os.path.join(output_dir, csv_filename)
    save_voting_results(approved, rejected, annotator_names, output_csv_path)
    
    print(f"완료: {filename} -> {output_json_path}")
    print(f"투표 결과: {output_csv_path}")
    print(f"  승인: {len(approved)}개, 기각: {len(rejected) - len(unmatched_polygons)}개, 미매칭: {len(unmatched_polygons)}개")

def merge_all_annotations(base_dir: str, output_dir: str, 
                         iou_threshold: float = 0.5, min_votes: int = 2):
    """
    모든 JSON 파일에 대해 병합을 수행한다.
    
    Args:
        base_dir: annotator 폴더들이 있는 기본 디렉토리
        output_dir: 결과 저장 디렉토리
        iou_threshold: IoU threshold
        min_votes: 최소 필요 투표 수
    """
    # 첫 번째 annotator 폴더에서 파일 목록 가져오기
    annotator_dirs = sorted([d for d in Path(base_dir).iterdir() if d.is_dir()])
    if not annotator_dirs:
        print("오류: annotator 폴더를 찾을 수 없습니다.")
        return
    
    first_dir = annotator_dirs[0]
    json_files = sorted([f.name for f in first_dir.glob('*.json')])
    
    print(f"총 {len(json_files)}개의 JSON 파일을 병합합니다.")
    
    for filename in json_files:
        try:
            merge_single_file(base_dir, filename, output_dir, iou_threshold, min_votes)
        except Exception as e:
            print(f"오류 발생 ({filename}): {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    # 사용 예시
    base_directory = '/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_inner/batch_3rd/batch_3rd_phase2_validation'
    output_directory = '/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_inner/batch_3rd/batch_3rd_phase2_validation/merged_annotations'
    
    merge_all_annotations(
        base_dir=base_directory,
        output_dir=output_directory,
        iou_threshold=0.5,
        min_votes=2
    )