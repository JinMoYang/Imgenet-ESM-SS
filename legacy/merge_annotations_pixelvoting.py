# merge_annotations.py

import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.optimize import linear_sum_assignment
import cv2


class AnnotationMerger:
    """
    여러 어노테이터의 instance segmentation 결과를 병합하는 클래스입니다.
    
    IoU 기반 매칭과 픽셀별 투표를 통해 최종 어노테이션을 생성합니다.
    description 필드의 속성들도 투표를 통해 보존합니다.
    각 객체별로 병합에 참여한 어노테이터 정보를 기록합니다.
    """
    
    def __init__(self, root_dir: str, iou_threshold: float = 0.5, min_annotators: int = 2):
        """
        Args:
            root_dir: annotator1, annotator2, annotator3 폴더를 포함하는 루트 디렉토리
            iou_threshold: 같은 객체로 간주할 최소 IoU 값
            min_annotators: 병합에 필요한 최소 어노테이터 수 (픽셀 투표 threshold)
        """
        self.root_dir = Path(root_dir)
        self.iou_threshold = iou_threshold
        self.min_annotators = min_annotators
        self.annotator_dirs = []
        self.excluded_objects = []
        self.merge_results = []
        
        # annotator 폴더 탐색
        for d in self.root_dir.iterdir():
            if d.is_dir() and d.name.startswith('annotator'):
                self.annotator_dirs.append(d)
        
        if len(self.annotator_dirs) == 0:
            raise ValueError(f"annotator 폴더를 찾을 수 없습니다: {root_dir}")
        
        print(f"발견된 annotator 폴더 수: {len(self.annotator_dirs)}")
    
    def parse_description(self, description: str) -> Dict[str, any]:
        """
        description 문자열에서 속성들을 파싱합니다.
        
        Args:
            description: "ishole, iscrowd=1" 같은 형태의 문자열
            
        Returns:
            파싱된 속성 딕셔너리 {'ishole': True, 'iscrowd': 1}
        """
        attributes = {}
        
        if not description or not isinstance(description, str):
            return attributes
        
        # 쉼표나 세미콜론으로 분리
        parts = description.replace(';', ',').split(',')
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # key=value 형태 (like iscrowd=1)
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip().lower()
                value = value.strip()
                
                # 숫자 변환 시도
                try:
                    if '.' in value:
                        attributes[key] = float(value)
                    else:
                        attributes[key] = int(value)
                except ValueError:
                    attributes[key] = value
            
            # 단순 플래그 (ishole)
            else:
                key = part.strip().lower()
                attributes[key] = True
        
        return attributes
    
    def vote_attributes(self, descriptions: List[str]) -> Dict[str, any]:
        """
        여러 description에서 추출한 속성들을 투표합니다.
        
        Args:
            descriptions: description 문자열 리스트
            
        Returns:
            과반수 이상 득표한 속성들의 딕셔너리
        """
        n = len(descriptions)
        if n == 0:
            return {}
        
        # 모든 속성 수집
        all_attributes = defaultdict(list)
        
        for desc in descriptions:
            attrs = self.parse_description(desc)
            for key, value in attrs.items():
                all_attributes[key].append(value)
        
        # 과반수 투표
        voted_attributes = {}
        threshold = n / 2.0
        
        for key, values in all_attributes.items():
            # boolean 플래그인 경우
            if all(isinstance(v, bool) for v in values):
                true_count = sum(values)
                if true_count > threshold:
                    voted_attributes[key] = True
            
            # 숫자 값인 경우
            elif all(isinstance(v, (int, float)) for v in values):
                # 각 값의 빈도 계산
                value_counts = Counter(values)
                most_common_value, count = value_counts.most_common(1)[0]
                
                if count > threshold:
                    voted_attributes[key] = most_common_value
            
            # 문자열 값인 경우
            else:
                value_counts = Counter(str(v) for v in values)
                most_common_value, count = value_counts.most_common(1)[0]
                
                if count > threshold:
                    voted_attributes[key] = most_common_value
        
        return voted_attributes
    
    def build_description(self, n_merged: int, voted_attributes: Dict[str, any]) -> str:
        """
        병합 정보와 투표된 속성들로 description을 구성합니다.
        
        Args:
            n_merged: 병합된 어노테이션 수
            voted_attributes: 투표로 선정된 속성들
            
        Returns:
            최종 description 문자열
        """
        parts = [f"merged from {n_merged} annotations"]
        
        # 속성들을 문자열로 변환
        for key, value in sorted(voted_attributes.items()):
            if isinstance(value, bool) and value:
                parts.append(key)
            elif not isinstance(value, bool):
                parts.append(f"{key}={value}")
        
        return ", ".join(parts)
    
    def compute_iou(self, poly1: List[List[float]], poly2: List[List[float]]) -> float:
        """
        두 폴리곤의 IoU를 계산합니다.
        
        Args:
            poly1, poly2: [[x1, y1], [x2, y2], ...] 형태의 좌표 리스트
            
        Returns:
            IoU 값 (0~1)
        """
        try:
            p1 = Polygon(poly1)
            p2 = Polygon(poly2)
            
            if not p1.is_valid:
                p1 = p1.buffer(0)
            if not p2.is_valid:
                p2 = p2.buffer(0)
            
            intersection = p1.intersection(p2).area
            union = p1.union(p2).area
            
            if union == 0:
                return 0.0
            
            return intersection / union
        except Exception as e:
            print(f"IoU 계산 중 오류 발생: {e}")
            return 0.0
    
    def merge_polygons_by_voting(
        self, 
        polygons: List[List[List[float]]], 
        image_height: int, 
        image_width: int,
        vote_threshold: int
    ) -> Optional[List[List[float]]]:
        """
        픽셀별 투표를 통해 폴리곤들을 병합합니다.
        
        Args:
            polygons: 폴리곤 리스트
            image_height: 이미지 높이
            image_width: 이미지 너비
            vote_threshold: 픽셀이 포함되기 위한 최소 투표 수
            
        Returns:
            병합된 폴리곤의 좌표 또는 None
        """
        try:
            # 투표 맵 생성
            vote_map = np.zeros((image_height, image_width), dtype=np.uint8)
            
            # 각 폴리곤을 마스크로 변환하여 투표
            for poly in polygons:
                mask = np.zeros((image_height, image_width), dtype=np.uint8)
                points = np.array(poly, dtype=np.int32)
                cv2.fillPoly(mask, [points], 1)
                vote_map += mask
            
            # threshold 적용
            merged_mask = (vote_map >= vote_threshold).astype(np.uint8)
            
            # 면적 체크
            if merged_mask.sum() < 10:
                return None
            
            # 마스크를 폴리곤으로 변환
            contours, _ = cv2.findContours(
                merged_mask, 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            if len(contours) == 0:
                return None
            
            # 가장 큰 contour 선택
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 면적이 너무 작으면 제외
            if cv2.contourArea(largest_contour) < 10:
                return None
            
            # contour를 폴리곤 좌표로 변환
            # shape: (n, 1, 2) -> (n, 2)
            polygon = largest_contour.squeeze().tolist()
            
            # 단일 점인 경우 처리
            if isinstance(polygon[0], int):
                return None
            
            # [[x, y], [x, y], ...] 형태로 변환
            result = [[float(pt[0]), float(pt[1])] for pt in polygon]
            
            return result
            
        except Exception as e:
            print(f"픽셀 투표 병합 중 오류 발생: {e}")
            return None
    
    def load_annotations(self) -> Dict[str, Dict[str, Dict]]:
        """
        모든 annotator의 JSON 파일을 로드합니다.
        
        Returns:
            {image_name: {annotator_name: annotation_data}}
        """
        annotations = defaultdict(dict)
        
        for annotator_dir in self.annotator_dirs:
            annotator_name = annotator_dir.name
            
            for json_file in annotator_dir.glob('*.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    image_name = data['imagePath']
                    annotations[image_name][annotator_name] = data
                    
                except Exception as e:
                    print(f"JSON 로드 실패 ({json_file}): {e}")
        
        print(f"로드된 이미지 수: {len(annotations)}")
        return dict(annotations)
    
    def match_objects(self, annotations: List[Dict]) -> List[List[Tuple[int, int]]]:
        """
        여러 어노테이션 간 객체를 매칭합니다.
        
        Args:
            annotations: 어노테이션 데이터 리스트
            
        Returns:
            매칭된 객체 인덱스 그룹 [(annotator_idx, object_idx), ...]
        """
        n_annotators = len(annotations)
        all_shapes = []
        shape_to_annotator = []
        
        # 모든 shape 수집
        for ann_idx, ann in enumerate(annotations):
            shapes = ann.get('shapes', [])
            all_shapes.extend(shapes)
            shape_to_annotator.extend([ann_idx] * len(shapes))
        
        if len(all_shapes) == 0:
            return []
        
        # IoU 행렬 계산
        n = len(all_shapes)
        iou_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # 같은 annotator의 shape는 매칭하지 않음
                if shape_to_annotator[i] == shape_to_annotator[j]:
                    continue
                
                iou = self.compute_iou(
                    all_shapes[i]['points'],
                    all_shapes[j]['points']
                )
                iou_matrix[i, j] = iou
                iou_matrix[j, i] = iou
        
        # 그래프 기반 클러스터링
        matched_groups = []
        visited = set()
        
        for i in range(n):
            if i in visited:
                continue
            
            group = [i]
            visited.add(i)
            
            # BFS로 연결된 노드 찾기
            queue = [i]
            while queue:
                curr = queue.pop(0)
                for j in range(n):
                    if j not in visited and iou_matrix[curr, j] >= self.iou_threshold:
                        # 같은 annotator가 아닌지 확인
                        if shape_to_annotator[j] not in [shape_to_annotator[k] for k in group]:
                            group.append(j)
                            visited.add(j)
                            queue.append(j)
            
            matched_groups.append(group)
        
        # 인덱스를 (annotator_idx, object_idx) 형태로 변환
        result_groups = []
        shape_counter = [0] * n_annotators
        
        for group in matched_groups:
            result_group = []
            for global_idx in group:
                ann_idx = shape_to_annotator[global_idx]
                # 해당 annotator 내에서의 object 인덱스 계산
                local_idx = global_idx - sum(len(annotations[i].get('shapes', [])) for i in range(ann_idx))
                result_group.append((ann_idx, local_idx))
            result_groups.append(result_group)
        
        return result_groups
    
    def merge_single_image(self, image_name: str, image_annotations: Dict[str, Dict]) -> Optional[Dict]:
        """
        하나의 이미지에 대한 어노테이션들을 병합합니다.
        
        Args:
            image_name: 이미지 파일명
            image_annotations: {annotator_name: annotation_data}
            
        Returns:
            병합된 어노테이션 또는 None
        """
        annotator_names = list(image_annotations.keys())
        annotations = list(image_annotations.values())
        
        if len(annotations) == 0:
            return None
        
        # 첫 번째 어노테이션을 템플릿으로 사용
        base_annotation = annotations[0].copy()
        base_annotation['shapes'] = []
        base_annotation['annotator'] = 'merged'
        
        # 이미지 크기 추출
        image_height = base_annotation.get('imageHeight')
        image_width = base_annotation.get('imageWidth')
        
        if image_height is None or image_width is None:
            print(f"경고: 이미지 크기 정보가 없습니다 ({image_name})")
            return None
        
        # 객체 매칭
        matched_groups = self.match_objects(annotations)
        
        for group in matched_groups:
            if len(group) < self.min_annotators:
                # 최소 어노테이터 수 미달
                for ann_idx, obj_idx in group:
                    shape = annotations[ann_idx]['shapes'][obj_idx]
                    annotator = annotations[ann_idx].get('annotator', annotator_names[ann_idx])
                    self.excluded_objects.append({
                        'image': image_name,
                        'label': shape['label'],
                        'annotator': annotator,
                        'reason': f'최소 어노테이터 수 미달 ({len(group)}/{self.min_annotators})'
                    })
                continue
            
            # 클래스명 투표
            labels = [annotations[ann_idx]['shapes'][obj_idx]['label'] for ann_idx, obj_idx in group]
            label_counts = Counter(labels)
            
            # 가장 많은 득표를 받은 레이블
            most_common_label, max_count = label_counts.most_common(1)[0]
            
            if max_count < self.min_annotators:
                # 클래스명 불일치
                for ann_idx, obj_idx in group:
                    shape = annotations[ann_idx]['shapes'][obj_idx]
                    annotator = annotations[ann_idx].get('annotator', annotator_names[ann_idx])
                    self.excluded_objects.append({
                        'image': image_name,
                        'label': shape['label'],
                        'annotator': annotator,
                        'reason': f'클래스명 불일치 (득표: {label_counts})'
                    })
                continue
            
            # 해당 레이블을 가진 폴리곤들과 description들만 선택
            selected_polygons = []
            selected_descriptions = []
            selected_annotators = []
            
            for ann_idx, obj_idx in group:
                shape = annotations[ann_idx]['shapes'][obj_idx]
                if shape['label'] == most_common_label:
                    selected_polygons.append(shape['points'])
                    selected_descriptions.append(shape.get('description', ''))
                    # JSON 파일의 annotator 필드 또는 폴더명 사용
                    annotator = annotations[ann_idx].get('annotator', annotator_names[ann_idx])
                    selected_annotators.append(annotator)
            
            # 픽셀별 투표로 폴리곤 병합
            merged_points = self.merge_polygons_by_voting(
                polygons=selected_polygons,
                image_height=image_height,
                image_width=image_width,
                vote_threshold=self.min_annotators
            )
            
            if merged_points is None:
                # 투표 결과 면적 부족
                for ann_idx, obj_idx in group:
                    shape = annotations[ann_idx]['shapes'][obj_idx]
                    if shape['label'] == most_common_label:
                        annotator = annotations[ann_idx].get('annotator', annotator_names[ann_idx])
                        self.excluded_objects.append({
                            'image': image_name,
                            'label': shape['label'],
                            'annotator': annotator,
                            'reason': '픽셀 투표 결과 면적 부족'
                        })
                continue
            
            # 속성 투표
            voted_attributes = self.vote_attributes(selected_descriptions)
            
            # description 구성
            final_description = self.build_description(
                n_merged=len(selected_polygons),
                voted_attributes=voted_attributes
            )
            
            # 병합된 shape 생성
            merged_shape = {
                'label': most_common_label,
                'score': None,
                'points': merged_points,
                'group_id': None,
                'description': final_description,
                'difficult': False,
                'shape_type': 'polygon',
                'flags': {},
                'attributes': {},
                'kie_linking': [],
                'annotators': selected_annotators  # 병합에 참여한 어노테이터 목록
            }
            
            base_annotation['shapes'].append(merged_shape)
        
        if len(base_annotation['shapes']) == 0:
            return None
        
        return base_annotation
    
    def merge_all(self, output_dir: str = 'merged_annotations'):
        """
        모든 어노테이션을 병합하고 결과를 저장합니다.
        
        Args:
            output_dir: 병합 결과를 저장할 디렉토리
        """
        output_path = self.root_dir / output_dir
        output_path.mkdir(exist_ok=True)
        
        print(f"병합 시작 (IoU threshold: {self.iou_threshold}, 최소 어노테이터: {self.min_annotators})")
        
        # 어노테이션 로드
        all_annotations = self.load_annotations()
        
        # 이미지별 병합
        for image_name, image_annotations in all_annotations.items():
            print(f"처리 중: {image_name}")
            
            merged = self.merge_single_image(image_name, image_annotations)
            
            if merged is not None:
                # JSON 저장
                output_file = output_path / f"{Path(image_name).stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(merged, f, indent=2, ensure_ascii=False)
                
                self.merge_results.append({
                    'image': image_name,
                    'status': 'success',
                    'n_objects': len(merged['shapes']),
                    'n_annotators': len(image_annotations)
                })
            else:
                self.merge_results.append({
                    'image': image_name,
                    'status': 'failed',
                    'n_objects': 0,
                    'n_annotators': len(image_annotations)
                })
        
        # 리포트 저장
        self.save_reports(output_path)
        
        print(f"\n병합 완료!")
        print(f"성공: {sum(1 for r in self.merge_results if r['status'] == 'success')}개")
        print(f"실패: {sum(1 for r in self.merge_results if r['status'] == 'failed')}개")
        print(f"제외된 객체: {len(self.excluded_objects)}개")
    
    def save_reports(self, output_path: Path):
        """
        병합 결과와 제외된 객체 정보를 CSV로 저장합니다.
        
        Args:
            output_path: 리포트를 저장할 디렉토리
        """
        # 병합 결과 리포트
        result_csv = output_path / 'merge_results.csv'
        with open(result_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image', 'status', 'n_objects', 'n_annotators'])
            writer.writeheader()
            writer.writerows(self.merge_results)
        
        print(f"병합 결과 저장: {result_csv}")
        
        # 제외된 객체 리포트
        excluded_csv = output_path / 'excluded_objects.csv'
        with open(excluded_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image', 'label', 'annotator', 'reason'])
            writer.writeheader()
            writer.writerows(self.excluded_objects)
        
        print(f"제외된 객체 정보 저장: {excluded_csv}")


def main():
    # 설정
    root_directory = '/Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st_phase2_validation'
    iou_threshold = 0.5
    min_annotators = 2
    
    # 병합 실행
    merger = AnnotationMerger(
        root_dir=root_directory,
        iou_threshold=iou_threshold,
        min_annotators=min_annotators
    )
    
    merger.merge_all(output_dir='merged_annotations')


if __name__ == '__main__':
    main()