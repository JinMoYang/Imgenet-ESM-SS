#!/usr/bin/env python3
# merge_annotations_v8.py

"""
MultiPolygon을 하나의 object로 취급하는 개선 버전
- MultiPolygon을 분리하지 않고 전체 object 단위로 clustering
- 각 어노테이터의 object는 최대 한 번만 하나의 cluster에 참여
- 2명 이상의 어노테이터가 참여한 cluster만 유효
"""

import argparse
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge annotations using IoU-based clustering (object-level)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='./annotations/',
        help='Input directory containing annotator subdirectories'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./merged_results/',
        help='Output directory for merged JSON and CSV'
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for clustering objects'
    )
    parser.add_argument(
        '--merge-method',
        type=str,
        choices=['union', 'intersection'],
        default='union',
        help='Method to merge polygons'
    )
    return parser.parse_args()


def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    else:
        return obj


def polygon_from_points(points: List[List[float]]) -> Polygon:
    if len(points) < 3:
        raise ValueError(f"Polygon must have at least 3 points, got {len(points)}")
    return Polygon(points)


def group_shapes_by_group_id(shapes: List[Dict]) -> List[Dict]:
    """shapes를 group_id 기준으로 그룹화한다."""
    from collections import OrderedDict
    
    grouped = OrderedDict()
    ungrouped = []
    
    for shape in shapes:
        group_id = shape.get('group_id')
        
        if group_id is None:
            ungrouped.append({
                'shapes': [shape],
                'group_id': None,
                'label': shape['label']
            })
        else:
            if group_id not in grouped:
                grouped[group_id] = {
                    'shapes': [],
                    'group_id': group_id,
                    'label': shape['label']
                }
            grouped[group_id]['shapes'].append(shape)
    
    result = list(grouped.values()) + ungrouped
    return result


def create_multipolygon_from_shapes(shapes: List[Dict]) -> MultiPolygon:
    """여러 shape을 하나의 MultiPolygon으로 결합한다."""
    polygons = []
    for shape in shapes:
        try:
            poly = polygon_from_points(shape['points'])
            polygons.append(poly)
        except Exception as e:
            print(f"Warning: Failed to convert shape to polygon: {e}")
            continue
    
    if len(polygons) == 0:
        raise ValueError("No valid polygons found")
    elif len(polygons) == 1:
        return polygons[0]
    else:
        return MultiPolygon(polygons)


def prepare_objects_for_clustering(
    grouped_shapes: List[Dict],
    annotator_idx: int
) -> List[Tuple]:
    """
    각 어노테이터의 object를 clustering을 위한 형태로 준비한다.
    
    Returns:
        List of (annotator_idx, object_idx, metadata, geometry)
    """
    objects = []
    
    for obj_idx, grouped_shape in enumerate(grouped_shapes):
        shapes = grouped_shape['shapes']
        label = grouped_shape['label']
        group_id = grouped_shape['group_id']
        
        try:
            # MultiPolygon 생성 (단일 polygon이면 Polygon 객체 반환)
            geometry = create_multipolygon_from_shapes(shapes)
            
            # Metadata 저장
            metadata = {
                'label': label,
                'group_id': group_id,
                'num_parts': len(shapes),
                'shapes': shapes,
                'description': shapes[0].get('description', '') if shapes else ''
            }
            
            objects.append((annotator_idx, obj_idx, metadata, geometry))
            
        except Exception as e:
            print(f"Warning: Failed to create geometry for object {obj_idx}: {e}")
            continue
    
    return objects


def calculate_iou(geom1, geom2) -> float:
    if not geom1.is_valid or not geom2.is_valid:
        return 0.0
    
    try:
        intersection = geom1.intersection(geom2).area
        union = geom1.union(geom2).area
        
        if union == 0:
            return 0.0
        
        return intersection / union
    except Exception as e:
        print(f"Warning: Error calculating IoU: {e}")
        return 0.0


def find_clusters_by_iou(
    all_objects: List[Tuple],
    iou_threshold: float
) -> List[Set[int]]:
    """
    IoU 기반 clustering을 수행한다.
    
    Args:
        all_objects: List of (annotator_idx, object_idx, metadata, geometry)
        iou_threshold: IoU threshold
    
    Returns:
        List of clusters (each cluster is a set of indices)
    """
    n = len(all_objects)
    
    # Build adjacency list
    adj = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            geom_i = all_objects[i][3]
            geom_j = all_objects[j][3]
            
            if geom_i is not None and geom_j is not None:
                iou = calculate_iou(geom_i, geom_j)
                if iou >= iou_threshold:
                    adj[i].add(j)
                    adj[j].add(i)
    
    # Find connected components using DFS
    visited = set()
    clusters = []
    
    def dfs(node, cluster):
        visited.add(node)
        cluster.add(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                dfs(neighbor, cluster)
    
    for i in range(n):
        if i not in visited:
            cluster = set()
            dfs(i, cluster)
            clusters.append(cluster)
    
    return clusters


def process_cluster(
    cluster_objects: List[Tuple],
    merge_method: str,
    annotator_names: List[str]
) -> Dict:
    """
    하나의 cluster를 처리하여 merged object를 생성한다.
    
    Args:
        cluster_objects: List of (annotator_idx, object_idx, metadata, geometry)
        merge_method: 'union' or 'intersection'
        annotator_names: List of annotator names
    
    Returns:
        Dict with processing result
    """
    # 각 어노테이터별로 object 그룹화
    annotator_objects = defaultdict(list)
    for obj in cluster_objects:
        ann_idx = obj[0]
        annotator_objects[ann_idx].append(obj)
    
    # 검증: 각 어노테이터는 최대 1개의 object만 기여해야 함
    for ann_idx, objs in annotator_objects.items():
        if len(objs) > 1:
            return {
                'valid': False,
                'reason': 'duplicate_annotator_contribution',
                'annotator': annotator_names[ann_idx]
            }
    
    # 최소 2명의 어노테이터가 참여해야 함
    if len(annotator_objects) < 2:
        return {
            'valid': False,
            'reason': 'insufficient_annotators'
        }
    
    # 각 어노테이터의 기여 추출
    contributions = {}
    geometries = []
    labels = []
    descriptions = []
    
    for ann_idx, objs in annotator_objects.items():
        # 각 어노테이터는 1개의 object만 기여함
        ann_idx_val, obj_idx_val, metadata, geometry = objs[0]
        
        contributions[ann_idx] = {
            'object_idx': obj_idx_val,
            'label': metadata['label'],
            'description': metadata['description'],
            'num_parts': metadata['num_parts'],
            'group_id': metadata['group_id']
        }
        
        geometries.append(geometry)
        labels.append(metadata['label'])
        descriptions.append(metadata['description'])
    
    # IoU 계산 (pairwise)
    ann_indices = list(contributions.keys())
    ious = []
    for i in range(len(ann_indices)):
        for j in range(i + 1, len(ann_indices)):
            idx_i = next(idx for idx, (a, _, _, _) in enumerate(cluster_objects) if a == ann_indices[i])
            idx_j = next(idx for idx, (a, _, _, _) in enumerate(cluster_objects) if a == ann_indices[j])
            geom_i = cluster_objects[idx_i][3]
            geom_j = cluster_objects[idx_j][3]
            iou = calculate_iou(geom_i, geom_j)
            ious.append(iou)
    
    min_iou = min(ious) if ious else 0.0
    
    # Label voting
    label_counts = Counter(labels)
    most_common = label_counts.most_common()
    
    num_annotators = len(ann_indices)
    
    if num_annotators == 2:
        if len(most_common) == 1:
            final_label = most_common[0][0]
            vote_status = 'unanimous'
        else:
            return {'valid': False, 'reason': 'label_disagreement'}
    else:  # num_annotators == 3
        if len(most_common) == 1:
            final_label = most_common[0][0]
            vote_status = 'unanimous'
        elif most_common[0][1] > most_common[1][1]:
            final_label = most_common[0][0]
            vote_status = 'voted'
        else:
            return {'valid': False, 'reason': 'label_tie'}
    
    # Description voting
    all_attributes = []
    for desc in descriptions:
        if desc.strip():
            attrs = {attr.strip() for attr in desc.split(',') if attr.strip()}
            all_attributes.append(attrs)
        else:
            all_attributes.append(set())
    
    unique_attrs = set()
    for attrs in all_attributes:
        unique_attrs.update(attrs)
    
    final_attrs = []
    if num_annotators == 2:
        for attr in unique_attrs:
            count = sum(1 for attrs in all_attributes if attr in attrs)
            if count == 2:
                final_attrs.append(attr)
    else:
        for attr in unique_attrs:
            count = sum(1 for attrs in all_attributes if attr in attrs)
            if count >= 2:
                final_attrs.append(attr)
    
    final_description = ', '.join(sorted(final_attrs)) if final_attrs else ''
    
    # Merge geometries
    merged_points_list = merge_polygons(geometries, merge_method)
    
    # Prepare annotator details
    annotator_details = []
    for ann_idx in ann_indices:
        contrib = contributions[ann_idx]
        annotator_details.append({
            'name': annotator_names[ann_idx],
            'label': contrib['label'],
            'description': contrib['description'],
            'object_index': contrib['object_idx'],
            'num_parts': contrib['num_parts'],
            'original_group_id': contrib['group_id']
        })
    
    return {
        'valid': True,
        'final_label': final_label,
        'final_description': final_description,
        'vote_status': vote_status,
        'min_iou': min_iou,
        'merged_points_list': merged_points_list,
        'num_annotators': num_annotators,
        'annotator_names': [annotator_names[idx] for idx in ann_indices],
        'annotator_labels': labels,
        'annotator_descriptions': descriptions,
        'annotator_details': annotator_details
    }


def merge_polygons_union(geometries: List) -> List[List[List[float]]]:
    flattened = []
    for geom in geometries:
        if isinstance(geom, MultiPolygon):
            flattened.extend(geom.geoms)
        else:
            flattened.append(geom)
    
    merged = unary_union(flattened)
    
    if merged.geom_type == 'MultiPolygon':
        result = []
        for geom in sorted(merged.geoms, key=lambda p: p.area, reverse=True):
            coords = list(geom.exterior.coords[:-1])
            result.append([[float(x), float(y)] for x, y in coords])
        return result
    else:
        coords = list(merged.exterior.coords[:-1])
        return [[[float(x), float(y)] for x, y in coords]]


def merge_polygons_intersection(geometries: List) -> List[List[List[float]]]:
    single_geoms = []
    for geom in geometries:
        if isinstance(geom, MultiPolygon):
            single_geoms.append(unary_union(geom))
        else:
            single_geoms.append(geom)
    
    merged = single_geoms[0]
    for geom in single_geoms[1:]:
        merged = merged.intersection(geom)
    
    if merged.is_empty or merged.geom_type not in ['Polygon', 'MultiPolygon']:
        merged = single_geoms[0]
    
    if merged.geom_type == 'MultiPolygon':
        result = []
        for geom in sorted(merged.geoms, key=lambda p: p.area, reverse=True):
            coords = list(geom.exterior.coords[:-1])
            result.append([[float(x), float(y)] for x, y in coords])
        return result
    else:
        coords = list(merged.exterior.coords[:-1])
        return [[[float(x), float(y)] for x, y in coords]]


def merge_polygons(geometries: List, method: str) -> List[List[List[float]]]:
    if method == 'union':
        return merge_polygons_union(geometries)
    elif method == 'intersection':
        return merge_polygons_intersection(geometries)
    else:
        raise ValueError(f"Unknown merge method: {method}")


def load_annotations_by_image(input_dir: str) -> Dict[str, List[Dict]]:
    input_path = Path(input_dir)
    annotations_by_image = defaultdict(list)
    
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    annotator_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    if not annotator_dirs:
        print(f"Error: No annotator subdirectories found in {input_dir}")
        sys.exit(1)
    
    print(f"\nFound {len(annotator_dirs)} annotator directories:")
    for d in annotator_dirs:
        print(f"  - {d.name}")
    
    total_files = 0
    for annotator_dir in annotator_dirs:
        json_files = list(annotator_dir.glob('*.json'))
        total_files += len(json_files)
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                annotator_name = data.get('annotator', annotator_dir.name)
                image_name = data.get('imagePath', json_file.stem)
                annotations_by_image[image_name].append({
                    'annotator': annotator_name,
                    'data': data,
                    'file': str(json_file)
                })
            except Exception as e:
                print(f"Warning: Failed to load {json_file}: {e}")
    
    print(f"\nLoaded {total_files} annotation files")
    print(f"Found {len(annotations_by_image)} unique images")
    
    return annotations_by_image


def process_image_annotations(
    image_name: str,
    annotations: List[Dict],
    iou_threshold: float,
    merge_method: str
) -> Tuple[Optional[Dict], Dict]:
    """
    이미지에 대한 어노테이션들을 처리하여 병합한다.
    
    Args:
        image_name: 이미지 이름
        annotations: 어노테이터들의 어노테이션 리스트 (2-3명)
        iou_threshold: IoU threshold
        merge_method: merge 방법
    
    Returns:
        (merged_annotation, report)
    """
    report = {
        'image_name': image_name,
        'mask_error': False,
        'class_error': False,
        'final_label': None,
        'vote_status': 'clean',
        'num_annotators': len(annotations),
        'num_clusters': 0,
        'num_objects_per_annotator': [],
        'annotators': [],
        'rejected_clusters': 0
    }
    
    if len(annotations) not in [2, 3]:
        report['mask_error'] = True
        report['vote_status'] = f'wrong_count_{len(annotations)}'
        return None, report
    
    annotator_names = [ann['annotator'] for ann in annotations]
    report['annotators'] = annotator_names
    
    # 각 어노테이터의 shapes를 group_id 기준으로 그룹화
    all_grouped_shapes = []
    for ann in annotations:
        shapes = ann['data'].get('shapes', [])
        grouped = group_shapes_by_group_id(shapes)
        all_grouped_shapes.append(grouped)
    
    report['num_objects_per_annotator'] = [len(g) for g in all_grouped_shapes]
    
    if any(len(g) == 0 for g in all_grouped_shapes):
        report['mask_error'] = True
        report['vote_status'] = 'no_shapes'
        return None, report
    
    # Object 준비 (MultiPolygon은 전체를 하나의 object로 취급)
    all_objects = []
    for ann_idx, grouped_shapes in enumerate(all_grouped_shapes):
        objects = prepare_objects_for_clustering(grouped_shapes, ann_idx)
        all_objects.extend(objects)
    
    print(f"  Image {image_name}: {report['num_objects_per_annotator']} objects → {len(all_objects)} total")
    
    # IoU 기반 clustering
    clusters = find_clusters_by_iou(all_objects, iou_threshold)
    
    # 각 cluster 처리
    merged_shapes = []
    rejected_count = 0
    
    for cluster_idx, cluster_indices in enumerate(clusters):
        cluster_objects = [all_objects[i] for i in cluster_indices]
        
        # Cluster 처리
        result = process_cluster(cluster_objects, merge_method, annotator_names)
        
        if not result['valid']:
            rejected_count += 1
            continue
        
        # Output shapes 생성
        merged_points_list = result['merged_points_list']
        
        # group_id 할당 (MultiPolygon인 경우)
        assigned_group_id = None
        if len(merged_points_list) > 1:
            assigned_group_id = cluster_idx + 10000
        
        # Shapes 생성
        for part_idx, merged_points in enumerate(merged_points_list):
            merged_shapes.append({
                'label': result['final_label'],
                'points': merged_points,
                'group_id': assigned_group_id,
                'shape_type': 'polygon',
                'flags': {},
                'description': result['final_description'],
                'score': None,
                'difficult': False,
                'attributes': {
                    'annotators': result['annotator_names'],
                    'annotator_labels': result['annotator_labels'],
                    'annotator_descriptions': result['annotator_descriptions'],
                    'annotator_details': result['annotator_details'],
                    'min_iou': float(round(result['min_iou'], 3)),
                    'vote_status': result['vote_status'],
                    'merge_info': f'Cluster {cluster_idx+1}: {result["num_annotators"]} annotators, part {part_idx+1}/{len(merged_points_list)}, merged using {merge_method}',
                    'is_multipart': len(merged_points_list) > 1,
                    'part_index': part_idx,
                    'total_parts': len(merged_points_list)
                },
                'kie_linking': []
            })
    
    report['num_clusters'] = len(clusters) - rejected_count
    report['num_shapes'] = len(merged_shapes)
    report['rejected_clusters'] = rejected_count
    
    if len(merged_shapes) == 0:
        report['mask_error'] = True
        report['vote_status'] = 'no_valid_clusters'
        return None, report
    
    # Report 업데이트
    all_labels = [s['label'] for s in merged_shapes]
    report['final_label'] = ','.join(set(all_labels))
    
    vote_statuses = [s['attributes']['vote_status'] for s in merged_shapes]
    if all(vs == 'unanimous' for vs in vote_statuses):
        report['vote_status'] = 'unanimous'
    elif any(vs == 'voted' for vs in vote_statuses):
        report['vote_status'] = 'voted'
    else:
        report['vote_status'] = 'clean'
    
    # Merged annotation 생성
    base_data = annotations[0]['data'].copy()
    image_height = base_data.get('imageHeight')
    image_width = base_data.get('imageWidth')
    
    merged_annotation = {
        'version': base_data.get('version', '5.0.0'),
        'flags': {},
        'shapes': merged_shapes,
        'imagePath': image_name,
        'imageData': None,
        'imageHeight': int(image_height) if image_height is not None else None,
        'imageWidth': int(image_width) if image_width is not None else None,
        'mergeMetadata': {
            'annotators': annotator_names,
            'mergeMethod': merge_method,
            'iouThreshold': float(iou_threshold),
            'mergeTimestamp': datetime.now().isoformat(),
            'totalClusters': int(len(clusters) - rejected_count),
            'totalShapes': int(len(merged_shapes)),
            'rejectedClusters': int(rejected_count),
            'objectCountsPerAnnotator': report['num_objects_per_annotator'],
            'algorithm': 'iou_clustering_v8_object_level'
        }
    }
    
    return merged_annotation, report


def main():
    args = parse_args()
    
    print("=" * 80)
    print("Annotation Merger v8 (Object-Level IoU Clustering)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input directory:  {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  IoU threshold:    {args.iou_threshold}")
    print(f"  Merge method:     {args.merge_method}")
    print(f"\nAlgorithm:")
    print(f"  MultiPolygon objects are treated as single units")
    print(f"  Clustering is performed at the object level")
    print(f"  Each annotator contributes at most one object per cluster")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    annotations_by_image = load_annotations_by_image(args.input_dir)
    
    print("\n" + "=" * 80)
    print("Processing images...")
    print("=" * 80)
    
    merged_annotations = []
    reports = []
    
    for image_name, annotations in annotations_by_image.items():
        merged, report = process_image_annotations(
            image_name,
            annotations,
            args.iou_threshold,
            args.merge_method
        )
        
        if merged is not None:
            merged_annotations.append(merged)
        
        reports.append(report)
        
        status = "✓" if merged is not None else "✗"
        obj_counts = report['num_objects_per_annotator']
        num_clusters = report.get('num_clusters', 0)
        print(f"  {status} {image_name}: {obj_counts} → {num_clusters} clusters ({report['vote_status']})")
    
    # Save merged annotations
    merged_dir = output_path / 'merged_annotations'
    merged_dir.mkdir(exist_ok=True)
    
    for merged in merged_annotations:
        image_name = merged['imagePath']
        base_name = Path(image_name).stem + '.json'
        output_file = merged_dir / base_name
        
        merged_clean = convert_numpy_types(merged)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_clean, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved {len(merged_annotations)} merged annotations to {merged_dir}/")
    
    # Save report CSV
    df = pd.DataFrame(reports)
    
    if 'annotators' in df.columns:
        df['annotators'] = df['annotators'].apply(lambda x: ','.join(x) if isinstance(x, list) else '')
    if 'num_objects_per_annotator' in df.columns:
        df['num_objects_per_annotator'] = df['num_objects_per_annotator'].apply(
            lambda x: ','.join(map(str, x)) if isinstance(x, list) else ''
        )
    
    csv_path = output_path / 'merge_report.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ Saved report to {csv_path}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"  Total images:           {len(reports)}")
    print(f"  Successfully merged:    {len(merged_annotations)}")
    print(f"  Failed:                 {len(reports) - len(merged_annotations)}")
    
    total_clusters = sum(r.get('num_clusters', 0) for r in reports)
    total_rejected = sum(r.get('rejected_clusters', 0) for r in reports)
    print(f"\n  Total clusters formed:  {total_clusters}")
    print(f"  Rejected clusters:      {total_rejected}")
    
    print("\n" + "=" * 80)
    print(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    main()


    """
    python merge_annotations_v8.py \
    --input-dir /Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st_validation \
    --output-dir /Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st_validation_output \
    --iou-threshold 0.2 \
    --merge-method union
    
    """