# polygon_merger.py

import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Union
from collections import defaultdict, Counter
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid
import cv2
import pandas as pd


class AnnotationMerger:
    """
    여러 annotator의 instance segmentation annotation을 병합하는 클래스.
    
    주요 기능:
    - 같은 group_id를 가진 폴리곤을 먼저 병합
    - 꼬인 폴리곤 자동 수정
    - IoU 기반 매칭
    - 클래스 및 속성 투표
    - 두 가지 병합 전략 (intersection, pixel_voting)
    - MultiPolygon을 여러 shape로 분할하여 저장
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        iou_threshold: float = 0.5,
        merge_strategy: str = "pixel_voting",
        annotators: List[str] = None,
        fix_invalid: bool = True
    ):
        """
        Args:
            input_dir: annotator별 폴더가 있는 상위 디렉토리 경로
            output_dir: 병합된 결과를 저장할 디렉토리 경로
            iou_threshold: 같은 객체로 판단할 IoU 임계값
            merge_strategy: 'intersection' 또는 'pixel_voting'
            annotators: annotator 폴더명 리스트
            fix_invalid: 꼬인 폴리곤 자동 수정 여부
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.iou_threshold = iou_threshold
        self.merge_strategy = merge_strategy
        self.annotators = annotators or ["annotator1", "annotator2", "annotator3"]
        self.fix_invalid = fix_invalid
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.merge_results = []
        self.excluded_objects = []
        
    def load_json(self, filepath: Path) -> Optional[Dict]:
        """JSON 파일을 로드합니다."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"파일 로드 실패: {filepath}, 오류: {e}")
            return None
    
    def save_json(self, data: Dict, filepath: Path):
        """JSON 파일을 저장합니다."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def fix_invalid_polygon(self, poly: Union[Polygon, MultiPolygon]) -> Union[Polygon, MultiPolygon, None]:
        """
        꼬이거나 유효하지 않은 폴리곤을 수정합니다.
        
        Args:
            poly: Shapely Polygon 또는 MultiPolygon 객체
            
        Returns:
            수정된 유효한 폴리곤 또는 None
        """
        if not self.fix_invalid:
            return poly
        
        try:
            if poly is None:
                return None
                
            if not poly.is_valid:
                # Shapely 2.0+에서는 make_valid 사용
                try:
                    poly = make_valid(poly)
                except:
                    # Shapely 1.x의 경우 buffer(0) 사용
                    poly = poly.buffer(0)
                
            # 빈 폴리곤이나 면적이 0인 경우 처리
            if poly.is_empty or poly.area == 0:
                return None
                
            return poly
        except Exception as e:
            print(f"폴리곤 수정 오류: {e}")
            return None
    
    def polygon_from_points(self, points: List[List[float]]) -> Optional[Polygon]:
        """
        좌표 리스트에서 Shapely Polygon을 생성하고 유효성을 검사합니다.
        
        Args:
            points: 폴리곤 좌표 리스트 [[x1, y1], [x2, y2], ...]
            
        Returns:
            유효한 Polygon 객체 또는 None
        """
        if not points or len(points) < 3:
            return None
        
        try:
            poly = Polygon(points)
            poly = self.fix_invalid_polygon(poly)
            return poly
        except Exception as e:
            print(f"폴리곤 생성 오류: {e}")
            return None
    
    def merge_shapes_by_group_id(
        self,
        shapes: List[Dict],
        annotator: str
    ) -> List[Dict]:
        """
        같은 group_id를 가진 shape들을 하나의 MultiPolygon으로 병합합니다.
        
        이는 IoU 매칭 전에 수행되며, 가려진 객체나 multi-part 객체를
        올바르게 처리하기 위한 사전 병합 단계입니다.
        
        Args:
            shapes: shape 딕셔너리 리스트
            annotator: annotator 이름
            
        Returns:
            병합된 shape 리스트 (각 shape는 geometry 필드를 포함)
        """
        # group_id별로 shape 그룹화
        grouped = defaultdict(list)
        no_group = []
        
        for idx, shape in enumerate(shapes):
            group_id = shape.get('group_id')
            if group_id is None:
                # group_id가 없는 경우 개별 처리
                no_group.append((idx, shape))
            else:
                grouped[group_id].append((idx, shape))
        
        merged_shapes = []
        
        # group_id가 있는 shape들 병합
        for group_id, shape_list in grouped.items():
            if len(shape_list) == 1:
                # 단일 shape는 그대로 유지
                idx, shape = shape_list[0]
                poly = self.polygon_from_points(shape['points'])
                if poly is not None:
                    merged_shapes.append({
                        'original_indices': [idx],
                        'shape': shape,
                        'geometry': poly,
                        'group_id': group_id,
                        'annotator': annotator
                    })
            else:
                # 여러 shape를 MultiPolygon으로 병합
                polygons = []
                labels = []
                descriptions = []
                original_indices = []
                
                for idx, shape in shape_list:
                    poly = self.polygon_from_points(shape['points'])
                    if poly is not None:
                        polygons.append(poly)
                        labels.append(shape['label'])
                        descriptions.append(shape.get('description', ''))
                        original_indices.append(idx)
                
                if not polygons:
                    continue
                
                # 모든 폴리곤을 하나의 MultiPolygon으로 통합
                try:
                    multi_poly = unary_union(polygons)
                    multi_poly = self.fix_invalid_polygon(multi_poly)
                    
                    if multi_poly is None or multi_poly.is_empty:
                        continue
                    
                    # 가장 많이 나온 label 선택
                    most_common_label = Counter(labels).most_common(1)[0][0]
                    
                    # description 병합
                    merged_description = ', '.join(filter(None, set(descriptions)))
                    
                    merged_shapes.append({
                        'original_indices': original_indices,
                        'shape': {
                            'label': most_common_label,
                            'points': [],
                            'group_id': group_id,
                            'description': merged_description,
                            'shape_type': 'polygon',
                            'difficult': False,
                            'flags': {},
                            'attributes': {},
                            'kie_linking': []
                        },
                        'geometry': multi_poly,
                        'group_id': group_id,
                        'annotator': annotator
                    })
                except Exception as e:
                    print(f"그룹 {group_id} 병합 오류: {e}")
                    continue
        
        # group_id가 없는 shape들 추가
        for idx, shape in no_group:
            poly = self.polygon_from_points(shape['points'])
            if poly is not None:
                merged_shapes.append({
                    'original_indices': [idx],
                    'shape': shape,
                    'geometry': poly,
                    'group_id': None,
                    'annotator': annotator
                })
        
        return merged_shapes
    
    def compute_geometry_iou(
        self,
        geom1: Union[Polygon, MultiPolygon],
        geom2: Union[Polygon, MultiPolygon]
    ) -> float:
        """
        두 기하 객체(Polygon 또는 MultiPolygon)의 IoU를 계산합니다.
        
        Args:
            geom1: 첫 번째 기하 객체
            geom2: 두 번째 기하 객체
            
        Returns:
            IoU 값 (0~1)
        """
        try:
            geom1 = self.fix_invalid_polygon(geom1)
            geom2 = self.fix_invalid_polygon(geom2)
            
            if geom1 is None or geom2 is None:
                return 0.0
            
            intersection_area = geom1.intersection(geom2).area
            union_area = geom1.union(geom2).area
            
            if union_area == 0:
                return 0.0
            
            return intersection_area / union_area
        except Exception as e:
            print(f"IoU 계산 오류: {e}")
            return 0.0
    
    def polygon_to_mask(
        self,
        geometry: Union[Polygon, MultiPolygon],
        width: int,
        height: int
    ) -> np.ndarray:
        """
        Shapely 기하 객체를 픽셀 마스크로 변환합니다.
        
        Args:
            geometry: Polygon 또는 MultiPolygon 객체
            width: 이미지 너비
            height: 이미지 높이
            
        Returns:
            이진 마스크 (height, width)
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        
        if isinstance(geometry, Polygon):
            polygons = [geometry]
        elif isinstance(geometry, MultiPolygon):
            polygons = list(geometry.geoms)
        else:
            return mask
        
        for poly in polygons:
            if poly.is_empty:
                continue
            coords = np.array(poly.exterior.coords, dtype=np.int32)
            cv2.fillPoly(mask, [coords], 1)
            
            # 내부 홀 처리
            for interior in poly.interiors:
                coords = np.array(interior.coords, dtype=np.int32)
                cv2.fillPoly(mask, [coords], 0)
        
        return mask
    
    def mask_to_polygon(self, mask: np.ndarray) -> List[List[float]]:
        """
        이진 마스크를 폴리곤으로 변환합니다.
        
        Args:
            mask: 이진 마스크
            
        Returns:
            폴리곤 좌표 리스트
        """
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return []
        
        # 가장 큰 contour 선택
        largest_contour = max(contours, key=cv2.contourArea)
        
        if len(largest_contour) < 3:
            return []
        
        polygon = largest_contour.squeeze()
        
        # 1차원 배열인 경우 처리
        if polygon.ndim == 1:
            return []
        
        return [[float(p[0]), float(p[1])] for p in polygon]
    
    def geometry_to_shapes(
        self,
        geometry: Union[Polygon, MultiPolygon],
        label: str,
        group_id: int,
        description: str
    ) -> List[Dict]:
        """
        Shapely 기하 객체를 shape 딕셔너리 리스트로 변환합니다.
        MultiPolygon인 경우 각 Polygon을 같은 group_id를 가진 별도의 shape로 생성합니다.
        
        Args:
            geometry: Polygon 또는 MultiPolygon 객체
            label: 클래스명
            group_id: 할당할 group_id
            description: description 문자열
            
        Returns:
            shape 딕셔너리 리스트
        """
        shapes = []
        
        # MultiPolygon인 경우 각 Polygon으로 분할
        if isinstance(geometry, MultiPolygon):
            polygons = list(geometry.geoms)
        elif isinstance(geometry, Polygon):
            polygons = [geometry]
        else:
            return shapes
        
        # 각 Polygon을 별도의 shape로 생성
        for poly in polygons:
            if poly.is_empty or poly.area == 0:
                continue
            
            coords = list(poly.exterior.coords[:-1])
            points = [[float(x), float(y)] for x, y in coords]
            
            if len(points) < 3:
                continue
            
            shape = {
                'label': label,
                'score': None,
                'points': points,
                'group_id': group_id,
                'description': description,
                'difficult': False,
                'shape_type': 'polygon',
                'flags': {},
                'attributes': {},
                'kie_linking': []
            }
            
            shapes.append(shape)
        
        return shapes
    
    def merge_by_intersection(
        self,
        geometries: List[Union[Polygon, MultiPolygon]]
    ) -> Union[Polygon, MultiPolygon, None]:
        """
        여러 기하 객체를 intersection으로 병합합니다.
        
        Args:
            geometries: Polygon 또는 MultiPolygon 객체 리스트
            
        Returns:
            병합된 기하 객체
        """
        try:
            valid_geoms = []
            for geom in geometries:
                geom = self.fix_invalid_polygon(geom)
                if geom is not None and not geom.is_empty:
                    valid_geoms.append(geom)
            
            if len(valid_geoms) == 0:
                return None
            
            # 교집합 계산
            intersection = valid_geoms[0]
            for geom in valid_geoms[1:]:
                intersection = intersection.intersection(geom)
            
            if intersection.is_empty or intersection.area == 0:
                return None
            
            return self.fix_invalid_polygon(intersection)
        except Exception as e:
            print(f"Intersection 병합 오류: {e}")
            return None
    
    def merge_by_pixel_voting(
        self,
        geometries: List[Union[Polygon, MultiPolygon]],
        width: int,
        height: int
    ) -> Union[Polygon, MultiPolygon, None]:
        """
        여러 기하 객체를 픽셀 단위 voting으로 병합합니다.
        과반수 이상이 포함시킨 픽셀만 최종 폴리곤에 포함됩니다.
        
        Args:
            geometries: Polygon 또는 MultiPolygon 객체 리스트
            width: 이미지 너비
            height: 이미지 높이
            
        Returns:
            병합된 기하 객체
        """
        vote_map = np.zeros((height, width), dtype=np.int32)
        
        for geom in geometries:
            mask = self.polygon_to_mask(geom, width, height)
            vote_map += mask
        
        # 과반수 기준
        threshold = len(geometries) / 2.0
        merged_mask = (vote_map > threshold).astype(np.uint8)
        
        # 마스크에서 모든 contour 추출 (여러 개일 수 있음)
        contours, _ = cv2.findContours(
            merged_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return None
        
        # 각 contour를 Polygon으로 변환
        polygons = []
        for contour in contours:
            if len(contour) < 3:
                continue
            
            polygon_points = contour.squeeze()
            if polygon_points.ndim == 1:
                continue
            
            points = [[float(p[0]), float(p[1])] for p in polygon_points]
            poly = self.polygon_from_points(points)
            
            if poly is not None and poly.area > 0:
                polygons.append(poly)
        
        if not polygons:
            return None
        
        # 여러 폴리곤이 있으면 MultiPolygon으로 반환
        if len(polygons) == 1:
            return polygons[0]
        else:
            return MultiPolygon(polygons)
    
    def extract_attributes(self, description: str) -> Dict[str, bool]:
        """
        description에서 ishole, iscrowd 속성을 추출합니다.
        
        Args:
            description: JSON의 description 필드
            
        Returns:
            {'ishole': bool, 'iscrowd': bool}
        """
        attributes = {'ishole': False, 'iscrowd': False}
        
        if not description:
            return attributes
        
        desc_lower = description.lower()
        if 'ishole' in desc_lower:
            attributes['ishole'] = True
        if 'iscrowd=1' in description or 'iscrowd' in desc_lower:
            attributes['iscrowd'] = True
        
        return attributes
    
    def vote_attributes(self, descriptions: List[str]) -> str:
        """
        여러 description에서 속성을 voting하여 최종 description을 생성합니다.
        
        Args:
            descriptions: description 문자열 리스트
            
        Returns:
            최종 description 문자열
        """
        ishole_count = 0
        iscrowd_count = 0
        
        for desc in descriptions:
            attrs = self.extract_attributes(desc)
            if attrs['ishole']:
                ishole_count += 1
            if attrs['iscrowd']:
                iscrowd_count += 1
        
        total = len(descriptions)
        result_parts = []
        
        if ishole_count > total / 2:
            result_parts.append('ishole')
        if iscrowd_count > total / 2:
            result_parts.append('iscrowd=1')
        
        return ', '.join(result_parts)
    
    def merge_annotations(
        self,
        image_name: str
    ) -> Tuple[Optional[Dict], List[Dict]]:
        """
        하나의 이미지에 대한 여러 annotator의 annotation을 병합합니다.
        
        처리 단계:
        1. 각 annotator의 JSON 로드
        2. group_id 기반 사전 병합 (각 annotator별)
        3. IoU 기반 크로스 매칭
        4. 클래스 및 속성 투표
        5. 폴리곤 병합
        6. MultiPolygon을 여러 shape로 분할
        7. 최종 group_id 할당
        
        Args:
            image_name: 이미지 파일명
            
        Returns:
            (병합된 JSON 데이터, 제외된 객체 정보 리스트)
        """
        # [1단계] 각 annotator의 JSON 로드
        annotations = {}
        for annotator in self.annotators:
            json_path = self.input_dir / annotator / image_name.replace(
                '.JPEG', '.json'
            ).replace('.jpg', '.json').replace('.png', '.json')
            
            if json_path.exists():
                annotations[annotator] = self.load_json(json_path)
        
        if not annotations:
            return None, []
        
        # 기준 데이터 설정
        base_data = next(iter(annotations.values()))
        width = base_data['imageWidth']
        height = base_data['imageHeight']
        
        # [2단계] 각 annotator별로 group_id 기반 병합 수행
        all_merged_shapes = []
        for annotator, data in annotations.items():
            if data is None:
                continue
            shapes = data.get('shapes', [])
            merged = self.merge_shapes_by_group_id(shapes, annotator)
            all_merged_shapes.extend(merged)
        
        # [3단계] IoU 기반 매칭
        matched_groups = []
        used_shapes = set()
        
        for i, shape_i in enumerate(all_merged_shapes):
            if i in used_shapes:
                continue
            
            # 현재 shape와 매칭되는 shape들 찾기
            current_group = {shape_i['annotator']: i}
            used_shapes.add(i)
            
            geom_i = shape_i.get('geometry')
            if geom_i is None:
                continue
            
            for j, shape_j in enumerate(all_merged_shapes[i+1:], start=i+1):
                if j in used_shapes:
                    continue
                if shape_j['annotator'] in current_group:
                    continue
                
                geom_j = shape_j.get('geometry')
                if geom_j is None:
                    continue
                
                iou = self.compute_geometry_iou(geom_i, geom_j)
                
                if iou >= self.iou_threshold:
                    current_group[shape_j['annotator']] = j
                    used_shapes.add(j)
            
            # 2명 이상이 매칭된 경우만 다음 단계로
            if len(current_group) >= 2:
                matched_groups.append(current_group)
        
        # [4~7단계] 병합된 shape 생성
        merged_shapes = []
        excluded = []
        
        # 각 이미지마다 1부터 시작하는 고유한 group_id 할당
        next_group_id = 1
        
        for matched_group in matched_groups:
            # [4단계] 클래스명, 기하 객체, description 수집
            labels = []
            geometries = []
            descriptions = []
            
            for annotator, shape_idx in matched_group.items():
                shape_data = all_merged_shapes[shape_idx]
                labels.append(shape_data['shape']['label'])
                
                geom = shape_data.get('geometry')
                if geom is not None:
                    geometries.append(geom)
                
                descriptions.append(shape_data['shape'].get('description', ''))
            
            if not geometries:
                continue
            
            # 과반수 클래스 결정
            label_counts = Counter(labels)
            most_common_label, count = label_counts.most_common(1)[0]
            
            if count < len(labels) / 2:
                # 과반수 미달
                for annotator, shape_idx in matched_group.items():
                    shape_data = all_merged_shapes[shape_idx]
                    excluded.append({
                        'image': image_name,
                        'object': shape_data['shape']['label'],
                        'annotator': annotator,
                        'reason': '과반수 클래스 합의 미달'
                    })
                continue
            
            # [5단계] 기하 객체 병합
            if self.merge_strategy == 'intersection':
                merged_geom = self.merge_by_intersection(geometries)
            else:  # pixel_voting
                merged_geom = self.merge_by_pixel_voting(geometries, width, height)
            
            if merged_geom is None or merged_geom.is_empty:
                for annotator, shape_idx in matched_group.items():
                    shape_data = all_merged_shapes[shape_idx]
                    excluded.append({
                        'image': image_name,
                        'object': shape_data['shape']['label'],
                        'annotator': annotator,
                        'reason': '병합 후 유효한 폴리곤 생성 실패'
                    })
                continue
            
            # 속성 병합
            merged_description = self.vote_attributes(descriptions)
            
            # [6~7단계] 최종 group_id 할당 및 MultiPolygon 분할
            final_group_id = next_group_id
            next_group_id += 1
            
            # MultiPolygon인 경우 각 Polygon을 같은 group_id로 분할하여 저장
            shapes_list = self.geometry_to_shapes(
                merged_geom,
                most_common_label,
                final_group_id,
                merged_description
            )
            
            if not shapes_list:
                for annotator, shape_idx in matched_group.items():
                    shape_data = all_merged_shapes[shape_idx]
                    excluded.append({
                        'image': image_name,
                        'object': shape_data['shape']['label'],
                        'annotator': annotator,
                        'reason': '병합 후 좌표 변환 실패'
                    })
                continue
            
            merged_shapes.extend(shapes_list)
        
        # 병합되지 않은 shape 기록
        for i, shape_data in enumerate(all_merged_shapes):
            if i not in used_shapes:
                excluded.append({
                    'image': image_name,
                    'object': shape_data['shape']['label'],
                    'annotator': shape_data['annotator'],
                    'reason': 'IoU 임계값 미달 또는 매칭 실패'
                })
        
        # 최종 JSON 구성
        merged_json = {
            'version': base_data.get('version', '3.2.6'),
            'flags': {},
            'shapes': merged_shapes,
            'imagePath': image_name,
            'imageData': None,
            'imageHeight': height,
            'imageWidth': width,
            'annotator': 'merged'
        }
        
        return merged_json, excluded
    
    def process_all(self):
        """
        모든 이미지에 대해 annotation을 병합합니다.
        """
        # 모든 이미지 파일명 수집
        all_images = set()
        for annotator in self.annotators:
            annotator_dir = self.input_dir / annotator
            if not annotator_dir.exists():
                print(f"경고: {annotator_dir} 폴더가 존재하지 않습니다.")
                continue
            
            for json_file in annotator_dir.glob('*.json'):
                data = self.load_json(json_file)
                if data and 'imagePath' in data:
                    all_images.add(data['imagePath'])
        
        if not all_images:
            print("처리할 이미지가 없습니다.")
            return
        
        print(f"총 {len(all_images)}개 이미지를 처리합니다.")
        print(f"IoU threshold: {self.iou_threshold}")
        print(f"병합 전략: {self.merge_strategy}")
        print(f"꼬인 폴리곤 수정: {self.fix_invalid}")
        print("-" * 50)
        
        for image_name in sorted(all_images):
            print(f"처리 중: {image_name}")
            merged_json, excluded = self.merge_annotations(image_name)
            
            if merged_json and len(merged_json['shapes']) > 0:
                output_path = self.output_dir / image_name.replace(
                    '.JPEG', '.json'
                ).replace('.jpg', '.json').replace('.png', '.json')
                
                self.save_json(merged_json, output_path)
                
                # group_id별 shape 개수 계산
                group_ids = set(s['group_id'] for s in merged_json['shapes'])
                
                self.merge_results.append({
                    'image': image_name,
                    'status': '성공',
                    'merged_objects': len(group_ids),
                    'total_shapes': len(merged_json['shapes'])
                })
                print(f"  → 성공: {len(group_ids)}개 객체 ({len(merged_json['shapes'])}개 shape)")
            else:
                self.merge_results.append({
                    'image': image_name,
                    'status': '실패',
                    'merged_objects': 0,
                    'total_shapes': 0
                })
                print(f"  → 실패: 병합된 객체 없음")
            
            if excluded:
                print(f"  → 제외: {len(excluded)}개 객체")
            
            self.excluded_objects.extend(excluded)
        
        # CSV 리포트 생성
        print("-" * 50)
        self.generate_report()
    
    def generate_report(self):
        """
        병합 결과를 CSV 파일로 생성합니다.
        """
        # 병합 결과 CSV
        if self.merge_results:
            results_df = pd.DataFrame(self.merge_results)
            results_path = self.output_dir / 'merge_results.csv'
            results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
            
            success_count = len(results_df[results_df['status'] == '성공'])
            fail_count = len(results_df[results_df['status'] == '실패'])
            total_objects = results_df['merged_objects'].sum()
            total_shapes = results_df['total_shapes'].sum()
            
            print(f"병합 결과 저장: {results_path}")
            print(f"  - 성공: {success_count}개 이미지")
            print(f"  - 실패: {fail_count}개 이미지")
            print(f"  - 총 병합된 객체: {total_objects}개")
            print(f"  - 총 생성된 shape: {total_shapes}개")
        
        # 제외된 객체 CSV
        if self.excluded_objects:
            excluded_df = pd.DataFrame(self.excluded_objects)
            excluded_path = self.output_dir / 'excluded_objects.csv'
            excluded_df.to_csv(excluded_path, index=False, encoding='utf-8-sig')
            print(f"제외된 객체 정보 저장: {excluded_path}")
            print(f"  - 총 제외된 객체: {len(self.excluded_objects)}개")


def main():
    """
    메인 실행 함수입니다.
    
    사용 예시:
        python polygon_merger.py
    """
    merger = AnnotationMerger(
        input_dir='/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st_phase2_validation',
        output_dir='/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st_phase2_validation/merged_annotations',
        iou_threshold=0.5,
        merge_strategy='pixel_voting',
        annotators=['annotator1', 'annotator2', 'annotator3'],
        fix_invalid=True
    )
    
    merger.process_all()
    print("\n병합 작업이 완료되었습니다.")


if __name__ == '__main__':
    main()