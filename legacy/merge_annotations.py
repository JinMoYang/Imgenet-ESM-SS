import os
import json
import csv
import shutil
from collections import Counter
from shapely.geometry import Polygon
from shapely.errors import ShapelyError

# --- 설정 변수 ---

# annotator1, annotator2, annotator3 폴더가 들어있는 상위 폴더
INPUT_BASE_DIR = '/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st_validation' 
# 병합된 json이 저장될 폴더 이름
OUTPUT_DIR = '/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st_validation_result'
# 로그 파일 이름
LOG_FILE = 'merge_log.csv'

DETAILED_LOG_FILE = 'merge_log_detailed.csv'

# 동일 객체로 판단할 IoU 임계값
IOU_THRESHOLD = 0.3
# 다수결로 인정할 최소 어노테이터 수 (예: 3명 중 2명)
MAJORITY_THRESHOLD = 2

# -------------------

def calculate_iou(poly1, poly2):
    """두 Shapely Polygon 객체의 IoU를 계산합니다."""
    try:
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
            
        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    except ShapelyError:
        return 0.0

def get_polygon_from_shape(shape):
    """JSON의 shape 딕셔너리에서 '유효한(valid)' Shapely Polygon 객체를 생성합니다."""
    try:
        points = shape['points']
        if len(points) < 3:
            return None
        
        # .buffer(0)을 적용하여 'invalid' 폴리곤(self-intersection 등)을 정리
        poly = Polygon(points).buffer(0) 
        
        # buffer(0) 결과가 Polygon이 아닐 수도 있음 (예: 비어있거나, 여러 조각)
        if poly.geom_type != 'Polygon':
            return None
            
        return poly
    except Exception:
        return None

def process_merging():
    """
    어노테이션 병합 작업을 수행합니다.
    """
    print(f"어노테이션 병합 시작... (기준 폴더: {INPUT_BASE_DIR})")
    
    # 출력 폴더 생성
    output_path = os.path.join(INPUT_BASE_DIR, OUTPUT_DIR)
    os.makedirs(output_path, exist_ok=True)
    
    # --- [수정됨] ---
    # 두 종류의 로그 리스트 초기화
    summary_log_data = [] # 파일별 요약 로그
    detailed_log_data = [] # 객체별 상세 로그 (실패/제외 내역)
    # -------------------
    
    annotator_dirs = [
        os.path.join(INPUT_BASE_DIR, d) 
        for d in ['annotator1', 'annotator2', 'annotator3'] 
        if os.path.isdir(os.path.join(INPUT_BASE_DIR, d))
    ]
    
    if not annotator_dirs:
        print("오류: 'annotator1', 'annotator2', 'annotator3' 폴더를 찾을 수 없습니다.")
        return

    # 모든 어노테이터 폴더를 스캔하여 고유한 JSON 파일 목록 생성
    all_json_files_set = set()
    for annotator_dir in annotator_dirs:
        try:
            files = [f for f in os.listdir(annotator_dir) if f.endswith('.json')]
            all_json_files_set.update(files)
        except FileNotFoundError:
            print(f"경고: '{annotator_dir}' 폴더를 찾을 수 없습니다. 건너뜁니다.")
            
    json_files = sorted(list(all_json_files_set))
    
    if not json_files:
        print("오류: 모든 어노테이터 폴더에서 JSON 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(json_files)}개의 고유한 JSON 파일을 처리합니다.")

    # 각 JSON 파일(이미지)에 대해 순회
    for json_filename in json_files:
        try:
            all_shapes_for_image = [] 
            loaded_jsons_data = [] 
            
            for annotator_dir in annotator_dirs:
                json_path = os.path.join(annotator_dir, json_filename)
                
                if os.path.exists(json_path):
                    annotator_name = os.path.basename(annotator_dir)
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if not loaded_jsons_data:
                                loaded_jsons_data.append(data)
                            
                            for shape in data.get('shapes', []):
                                poly = get_polygon_from_shape(shape)
                                if poly:
                                    all_shapes_for_image.append({
                                        'poly': poly,
                                        'label': shape['label'],
                                        'annotator': annotator_name,
                                        'original_shape': shape
                                    })
                    except Exception as e:
                        print(f"경고: {json_path} 파일 로드/파싱 오류: {e}")


            if not loaded_jsons_data:
                summary_log_data.append([json_filename, 'failed', 'JSON 파일을 로드할 수 없음 (파일이 손상되었을 수 있습니다)'])
                continue
                
            template_data = loaded_jsons_data[0]
            
            # --- 1. IoU 기반 객체 그룹핑 ---
            num_shapes = len(all_shapes_for_image)
            visited = [False] * num_shapes
            groups = [] 
            
            for i in range(num_shapes):
                if not visited[i]:
                    group = []
                    stack = [i]
                    visited[i] = True
                    
                    while stack:
                        node_idx = stack.pop()
                        group.append(all_shapes_for_image[node_idx])
                        
                        for neighbor_idx in range(num_shapes):
                            if not visited[neighbor_idx]:
                                iou = calculate_iou(
                                    all_shapes_for_image[node_idx]['poly'],
                                    all_shapes_for_image[neighbor_idx]['poly']
                                )
                                if iou > IOU_THRESHOLD:
                                    visited[neighbor_idx] = True
                                    stack.append(neighbor_idx)
                    groups.append(group)
            
            # --- 2. 그룹별 다수결 투표 및 폴리곤 교집합 계산 (상세 로깅 추가) ---
            final_merged_shapes = []
            
            for group in groups:
                labels = [s['label'] for s in group]
                label_counts = Counter(labels)
                
                final_label = None
                polys_to_intersect = []

                # 다수결 확인
                for label, count in label_counts.items():
                    if count >= MAJORITY_THRESHOLD:
                        final_label = label
                        polys_to_intersect = [s['poly'] for s in group if s['label'] == label]
                        break
                
                # --- [수정됨] 상세 로그 기록 로직 ---
                if final_label:
                    # [사유: 소수 의견]
                    # 다수결(final_label)은 있으나, 다른 레이블을 가진 객체들 기록
                    for shape_info in group:
                        if shape_info['label'] != final_label:
                            detailed_log_data.append([
                                json_filename,
                                shape_info['annotator'],
                                shape_info['label'],
                                f'Minority Label (Group decision was {final_label})'
                            ])
                    
                    # 교집합 연산
                    merged_poly = polys_to_intersect[0]
                    for p in polys_to_intersect[1:]:
                        merged_poly = merged_poly.intersection(p)
                        
                    if not merged_poly.is_empty and merged_poly.geom_type == 'Polygon':
                        # [성공] 교집합 성공, 병합 객체 생성
                        coords = list(merged_poly.exterior.coords)
                        final_points = [[coord[0], coord[1]] for coord in coords[:-1]]

                        if len(final_points) >= 3:
                            new_shape = group[0]['original_shape'].copy()
                            new_shape['label'] = final_label
                            new_shape['points'] = final_points
                            new_shape['group_id'] = None
                            new_shape['description'] = f"Merged from {len(set(s['annotator'] for s in group if s['label'] == final_label))} annotators"
                            new_shape['attributes'] = {}
                            new_shape['flags'] = {}
                            final_merged_shapes.append(new_shape)
                        else:
                            # [사유: 교집합 실패] 교집합 결과가 유효하지 않은 폴리곤(점이 3개 미만)
                            for shape_info in group:
                                if shape_info['label'] == final_label:
                                    detailed_log_data.append([
                                        json_filename,
                                        shape_info['annotator'],
                                        shape_info['label'],
                                        'Intersection Failed (Resulting polygon < 3 points)'
                                    ])
                    else:
                        # [사유: 교집합 실패] 교집합 결과가 비어있거나 MultiPolygon인 경우
                        for shape_info in group:
                            if shape_info['label'] == final_label:
                                detailed_log_data.append([
                                    json_filename,
                                    shape_info['annotator'],
                                    shape_info['label'],
                                    'Intersection Failed (Empty or invalid geometry)'
                                ])
                else:
                    # [사유: 다수결 실패]
                    # 그룹 내 다수결 레이블이 없음 (모두 1표씩이거나, 2표 이상이 없는 경우)
                    for shape_info in group:
                        detailed_log_data.append([
                            json_filename,
                            shape_info['annotator'],
                            shape_info['label'],
                            f'No Majority (Group labels: {dict(label_counts)})'
                        ])
                # --- [수정 완료] ---

            # --- 3. 최종 병합 JSON 파일 생성 ---
            merged_data = {
                "version": template_data.get("version", "merged"),
                "flags": {},
                "shapes": final_merged_shapes,
                "imagePath": template_data["imagePath"],
                "imageData": None,
                "imageHeight": template_data["imageHeight"],
                "imageWidth": template_data["imageWidth"],
                "annotator": "merged_v3"
            }
            
            output_json_path = os.path.join(output_path, json_filename)
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)
                
            summary_log_data.append([json_filename, 'success', f'Merged {len(final_merged_shapes)} objects.'])

        except Exception as e:
            summary_log_data.append([json_filename, 'failed', str(e)])
            print(f"오류 발생 ({json_filename}): {e}")

    # --- 4. CSV 로그 파일 저장 (요약 / 상세) ---
    
    # 4-1. 요약 로그 저장
    log_file_path = os.path.join(INPUT_BASE_DIR, LOG_FILE)
    try:
        with open(log_file_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['json_filename', 'status', 'message'])
            writer.writerows(summary_log_data)
        print(f"\n작업 완료. 병합된 파일은 '{output_path}'에 저장되었습니다.")
        print(f"요약 로그 파일은 '{log_file_path}'에 저장되었습니다.")
    except IOError as e:
        print(f"오류: 요약 로그 파일 저장 실패 - {e}")

    # 4-2. 상세 로그 저장 (제외 내역)
    detailed_log_file_path = os.path.join(INPUT_BASE_DIR, DETAILED_LOG_FILE)
    try:
        with open(detailed_log_file_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['json_filename', 'annotator', 'label', 'rejection_reason'])
            writer.writerows(detailed_log_data)
        print(f"제외 객체 상세 로그 파일은 '{detailed_log_file_path}'에 저장되었습니다.")
    except IOError as e:
        print(f"오류: 상세 로그 파일 저장 실패 - {e}")


# 스크립트 실행
if __name__ == "__main__":
    if INPUT_BASE_DIR == 'path/to/your/annotations':
        print("="*50)
        print(" 경고: 'INPUT_BASE_DIR' 변수를 실제 경로로 수정해주세요.")
        print("="*50)
    else:
        process_merging()