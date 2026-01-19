# merge_annotations.py

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set
import numpy as np
from PIL import Image, ImageDraw
import cv2
from collections import defaultdict, Counter


class UnionFind:
    """Union-Find 자료구조로 폴리곤 ID 클러스터링을 관리합니다."""
    
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def make_set(self, x):
        """새로운 집합을 생성합니다."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
    
    def find(self, x):
        """x가 속한 집합의 대표 원소를 찾습니다."""
        if x not in self.parent:
            self.make_set(x)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """두 집합을 합칩니다."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
    
    def get_clusters(self) -> Dict[str, List[str]]:
        """클러스터별로 그룹화된 ID 목록을 반환합니다."""
        clusters = defaultdict(list)
        for node in self.parent.keys():
            root = self.find(node)
            clusters[root].append(node)
        return dict(clusters)


def parse_description(description: str) -> Set[str]:
    """description 문자열에서 속성들을 추출합니다.
    
    Args:
        description: "ishole, iscrowd=1" 같은 형태의 문자열
    
    Returns:
        추출된 속성들의 집합 (예: {'ishole', 'iscrowd=1'})
    """
    if not description:
        return set()
    
    # 대소문자 구분 없이 처리
    description_lower = description.lower()
    
    # ishole, iscrowd=1, iscrowd=0 같은 패턴 추출
    # 단어 경계를 고려하여 추출
    attributes = set()
    
    # ishole 패턴 찾기
    if re.search(r'\bishole\b', description_lower):
        attributes.add('ishole')
    
    # iscrowd=숫자 패턴 찾기
    crowd_match = re.search(r'\biscrowd\s*=\s*(\d+)\b', description_lower)
    if crowd_match:
        attributes.add(f'iscrowd={crowd_match.group(1)}')
    
    return attributes


def format_description(attributes: Set[str]) -> str:
    """속성 집합을 description 문자열로 변환합니다.
    
    Args:
        attributes: 속성들의 집합
    
    Returns:
        포맷된 description 문자열
    """
    if not attributes:
        return ""
    
    # 정렬하여 일관된 순서 유지
    sorted_attrs = sorted(attributes)
    return ", ".join(sorted_attrs)


def polygon_to_mask(points: List[List[float]], height: int, width: int) -> np.ndarray:
    """폴리곤 좌표를 binary mask로 변환합니다.
    
    Args:
        points: [[x1, y1], [x2, y2], ...] 형태의 폴리곤 좌표
        height: 이미지 높이
        width: 이미지 너비
    
    Returns:
        (height, width) 크기의 binary mask
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    points_array = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points_array], 1)
    return mask


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """두 binary mask 간 IoU를 계산합니다."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    if union == 0:
        return 0.0
    
    return float(intersection) / float(union)


def merge_masks_by_group(shapes: List[dict], height: int, width: int) -> Dict[int, np.ndarray]:
    """group_id가 동일한 폴리곤들을 합쳐서 mask를 생성합니다.
    
    Args:
        shapes: JSON의 shapes 리스트
        height: 이미지 높이
        width: 이미지 너비
    
    Returns:
        {group_id: merged_mask} 딕셔너리
    """
    group_masks = defaultdict(lambda: np.zeros((height, width), dtype=np.uint8))
    
    for shape in shapes:
        group_id = shape.get('group_id')
        if group_id is not None:
            mask = polygon_to_mask(shape['points'], height, width)
            group_masks[group_id] = np.logical_or(group_masks[group_id], mask).astype(np.uint8)
    
    return dict(group_masks)


def load_annotations(base_dir: str, annotators: List[str]) -> Dict[str, Dict[str, dict]]:
    """모든 annotator의 annotation을 로드합니다.
    
    Args:
        base_dir: 기본 디렉토리 경로
        annotators: annotator 폴더명 리스트 (예: ['annotator1', 'annotator2', 'annotator3'])
    
    Returns:
        {annotator: {image_name: json_data}} 형태의 중첩 딕셔너리
    """
    annotations = {}
    
    for annotator in annotators:
        annotator_path = Path(base_dir) / annotator
        annotations[annotator] = {}
        
        for json_file in annotator_path.glob('*.json'):
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                image_name = data['imagePath']
                annotations[annotator][image_name] = data
    
    return annotations


def assign_polygon_ids(annotations: Dict[str, Dict[str, dict]]) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """각 폴리곤에 고유 ID를 할당합니다.
    
    Returns:
        polygon_data: {polygon_id: {'annotator': ..., 'image': ..., 'shape': ..., 'mask': ...}}
        id_to_image: {polygon_id: image_name}
    """
    polygon_data = {}
    id_to_image = {}
    
    for annotator, images in annotations.items():
        for image_name, data in images.items():
            height = data['imageHeight']
            width = data['imageWidth']
            
            for idx, shape in enumerate(data['shapes']):
                polygon_id = f"{annotator}_{image_name}_{idx}"
                mask = polygon_to_mask(shape['points'], height, width)
                
                polygon_data[polygon_id] = {
                    'annotator': annotator,
                    'image': image_name,
                    'shape': shape,
                    'mask': mask,
                    'height': height,
                    'width': width
                }
                id_to_image[polygon_id] = image_name
    
    return polygon_data, id_to_image


def match_polygons(polygon_data: Dict[str, dict], id_to_image: Dict[str, str], 
                   iou_threshold: float = 0.5) -> UnionFind:
    """IoU 기반으로 폴리곤들을 매칭합니다.
    
    Args:
        polygon_data: 폴리곤 정보 딕셔너리
        id_to_image: polygon_id to image_name 매핑
        iou_threshold: IoU threshold (기본값 0.5)
    
    Returns:
        매칭된 폴리곤 클러스터를 관리하는 UnionFind 객체
    """
    uf = UnionFind()
    
    # 모든 polygon_id를 집합으로 초기화
    for polygon_id in polygon_data.keys():
        uf.make_set(polygon_id)
    
    # 이미지별로 그룹화
    image_to_polygons = defaultdict(list)
    for polygon_id, image_name in id_to_image.items():
        image_to_polygons[image_name].append(polygon_id)
    
    # 같은 이미지 내에서만 비교
    for image_name, polygon_ids in image_to_polygons.items():
        # group_id별로 합쳐진 mask 생성 (각 annotator별로)
        annotator_group_masks = {}
        
        for polygon_id in polygon_ids:
            data = polygon_data[polygon_id]
            annotator = data['annotator']
            group_id = data['shape'].get('group_id')
            
            if group_id is not None:
                key = (annotator, group_id)
                if key not in annotator_group_masks:
                    annotator_group_masks[key] = {
                        'mask': np.zeros_like(data['mask']),
                        'polygon_ids': []
                    }
                annotator_group_masks[key]['mask'] = np.logical_or(
                    annotator_group_masks[key]['mask'],
                    data['mask']
                ).astype(np.uint8)
                annotator_group_masks[key]['polygon_ids'].append(polygon_id)
        
        # 모든 폴리곤 쌍에 대해 IoU 계산
        for i, id_a in enumerate(polygon_ids):
            for id_b in polygon_ids[i+1:]:
                data_a = polygon_data[id_a]
                data_b = polygon_data[id_b]
                
                # 같은 annotator의 폴리곤은 비교하지 않음
                if data_a['annotator'] == data_b['annotator']:
                    continue
                
                # mask 준비
                mask_a = data_a['mask']
                mask_b = data_b['mask']
                
                # group_id 확인
                group_a = data_a['shape'].get('group_id')
                group_b = data_b['shape'].get('group_id')
                
                # group_id가 있으면 합쳐진 mask 사용
                if group_a is not None:
                    key_a = (data_a['annotator'], group_a)
                    if key_a in annotator_group_masks:
                        mask_a = annotator_group_masks[key_a]['mask']
                
                if group_b is not None:
                    key_b = (data_b['annotator'], group_b)
                    if key_b in annotator_group_masks:
                        mask_b = annotator_group_masks[key_b]['mask']
                
                # IoU 계산 및 매칭
                iou = compute_iou(mask_a, mask_b)
                
                if iou >= iou_threshold:
                    # group에 속한 모든 polygon들을 union
                    if group_a is not None:
                        key_a = (data_a['annotator'], group_a)
                        if key_a in annotator_group_masks:
                            for pid in annotator_group_masks[key_a]['polygon_ids']:
                                uf.union(id_a, pid)
                    
                    if group_b is not None:
                        key_b = (data_b['annotator'], group_b)
                        if key_b in annotator_group_masks:
                            for pid in annotator_group_masks[key_b]['polygon_ids']:
                                uf.union(id_b, pid)
                    
                    uf.union(id_a, id_b)
    
    return uf


def majority_voting(cluster_ids: List[str], polygon_data: Dict[str, dict]) -> Tuple[np.ndarray, str, str]:
    """클러스터 내 폴리곤들에 대해 majority voting을 수행합니다.
    
    Args:
        cluster_ids: 같은 클러스터에 속한 polygon_id 리스트
        polygon_data: 폴리곤 정보 딕셔너리
    
    Returns:
        (final_mask, final_label, final_description) 튜플
    """
    if not cluster_ids:
        return None, None, None
    
    # 이미지 크기 확인
    sample_data = polygon_data[cluster_ids[0]]
    height = sample_data['height']
    width = sample_data['width']
    
    # 픽셀별 투표
    vote_mask = np.zeros((height, width), dtype=np.int32)
    labels = []
    all_attributes = []
    
    for polygon_id in cluster_ids:
        data = polygon_data[polygon_id]
        vote_mask += data['mask']
        labels.append(data['shape']['label'])
        
        # description 파싱
        description = data['shape'].get('description', '')
        attributes = parse_description(description)
        all_attributes.append(attributes)
    
    # 과반수 이상이 foreground인 픽셀만 선택
    threshold = len(cluster_ids) / 2.0
    final_mask = (vote_mask > threshold).astype(np.uint8)
    
    # 레이블 투표
    label_counter = Counter(labels)
    final_label = label_counter.most_common(1)[0][0]
    
    # 속성 투표 (각 속성별로 과반수 이상이면 포함)
    attribute_votes = defaultdict(int)
    for attrs in all_attributes:
        for attr in attrs:
            attribute_votes[attr] += 1
    
    final_attributes = set()
    attr_threshold = len(cluster_ids) / 2.0
    for attr, count in attribute_votes.items():
        if count > attr_threshold:
            final_attributes.add(attr)
    
    final_description = format_description(final_attributes)
    
    return final_mask, final_label, final_description


def mask_to_polygon(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    """binary mask를 폴리곤 좌표 리스트로 변환합니다.
    
    Returns:
        각 connected component에 대한 폴리곤 좌표 리스트
    """
    # connected components 찾기
    num_labels, labels = cv2.connectedComponents(mask)
    
    polygons = []
    for label_id in range(1, num_labels):
        component_mask = (labels == label_id).astype(np.uint8)
        
        # contour 추출
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) >= 3:  # 최소 3개 점 필요
                polygon = contour.squeeze().tolist()
                # [[x, y], ...] 형태로 변환
                if len(polygon) > 0:
                    if isinstance(polygon[0], int):
                        polygon = [polygon]
                    polygons.append([[float(p[0]), float(p[1])] for p in polygon])
    
    return polygons


def merge_annotations_workflow(base_dir: str, annotators: List[str], 
                                output_path: str, iou_threshold: float = 0.5):
    """전체 annotation 병합 작업을 수행합니다.
    
    Args:
        base_dir: annotator 폴더들이 있는 기본 디렉토리
        annotators: annotator 폴더명 리스트
        output_path: 결과를 저장할 디렉토리
        iou_threshold: IoU threshold
    """
    print(f"[1/5] annotation 파일을 로드합니다...")
    annotations = load_annotations(base_dir, annotators)
    
    print(f"[2/5] 각 폴리곤에 고유 ID를 할당합니다...")
    polygon_data, id_to_image = assign_polygon_ids(annotations)
    
    print(f"[3/5] IoU 기반으로 폴리곤을 매칭합니다... (threshold={iou_threshold})")
    uf = match_polygons(polygon_data, id_to_image, iou_threshold)
    
    print(f"[4/5] majority voting을 수행합니다...")
    clusters = uf.get_clusters()
    
    # 이미지별로 결과 정리
    image_results = defaultdict(list)
    
    for cluster_id, polygon_ids in clusters.items():
        if not polygon_ids:
            continue
        
        # majority voting (레이블 + description 속성)
        final_mask, final_label, final_description = majority_voting(polygon_ids, polygon_data)
        
        if final_mask is None or final_mask.sum() == 0:
            continue
        
        # mask를 폴리곤으로 변환
        polygons = mask_to_polygon(final_mask)
        
        # 이미지 이름 확인
        image_name = id_to_image[polygon_ids[0]]
        
        # 여러 connected component가 있으면 group_id 할당
        if len(polygons) > 1:
            for polygon in polygons:
                shape = {
                    'label': final_label,
                    'points': polygon,
                    'group_id': cluster_id,
                    'description': final_description,
                    'shape_type': 'polygon',
                    'flags': {},
                    'score': None,
                    'difficult': False,
                    'attributes': {},
                    'kie_linking': []
                }
                image_results[image_name].append(shape)
        elif len(polygons) == 1:
            shape = {
                'label': final_label,
                'points': polygons[0],
                'group_id': None,
                'description': final_description,
                'shape_type': 'polygon',
                'flags': {},
                'score': None,
                'difficult': False,
                'attributes': {},
                'kie_linking': []
            }
            image_results[image_name].append(shape)
    
    print(f"[5/5] 결과를 저장합니다...")
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for image_name, shapes in image_results.items():
        # 원본 데이터에서 메타정보 가져오기
        sample_annotation = None
        for annotator in annotators:
            if image_name in annotations[annotator]:
                sample_annotation = annotations[annotator][image_name]
                break
        
        if sample_annotation is None:
            continue
        
        result_json = {
            'version': sample_annotation['version'],
            'flags': {},
            'shapes': shapes,
            'imagePath': image_name,
            'imageData': None,
            'imageHeight': sample_annotation['imageHeight'],
            'imageWidth': sample_annotation['imageWidth']
        }
        
        # JSON 파일명에서 확장자를 제외한 부분 추출
        json_filename = Path(image_name).stem + '.json'
        output_file = output_dir / json_filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)
    
    print(f"완료되었습니다. 총 {len(image_results)}개 이미지의 결과가 {output_path}에 저장되었습니다.")


if __name__ == '__main__':
    # 사용 예시
    base_directory = '/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st_validation'  # annotator1, annotator2, annotator3 폴더가 있는 경로
    annotator_list = ['annotator1', 'annotator2', 'annotator3']
    output_directory = '/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st_validation/merged_annotations'
    iou_thresh = 0.5
    
    merge_annotations_workflow(base_directory, annotator_list, output_directory, iou_thresh)