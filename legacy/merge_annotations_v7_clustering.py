#!/usr/bin/env python3
# merge_annotations_v7_clustering_fixed.py

"""
MultiPolygon 처리 개선 버전
- MultiPolygon의 각 part를 독립적인 객체로 분리하여 clustering
- Clustering 후 원본 group_id 기반으로 재결합
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
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Merge annotations using IoU-based clustering (MultiPolygon fixed)'
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
    """Recursively convert numpy types to Python native types."""
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
    """Convert list of points to Shapely Polygon."""
    if len(points) < 3:
        raise ValueError(f"Polygon must have at least 3 points, got {len(points)}")
    return Polygon(points)


def group_shapes_by_group_id(shapes: List[Dict]) -> List[Dict]:
    """Group shapes by their group_id."""
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


def explode_multipolygon_objects(grouped_shapes: List[Dict], annotator_idx: int) -> List[Tuple]:
    """
    MultiPolygon의 각 part를 독립적인 객체로 분리한다.
    
    Args:
        grouped_shapes: group_id로 그룹화된 shapes 리스트
        annotator_idx: 현재 어노테이터의 인덱스
    
    Returns:
        List of (annotator_idx, object_idx, part_idx, shape_dict, polygon, original_group_id)
    """
    exploded_objects = []
    
    for obj_idx, grouped_shape in enumerate(grouped_shapes):
        shapes = grouped_shape['shapes']
        original_group_id = grouped_shape['group_id']
        label = grouped_shape['label']
        
        # 각 shape(part)를 독립적인 객체로 추가
        for part_idx, shape in enumerate(shapes):
            try:
                poly = polygon_from_points(shape['points'])
                
                # 원본 shape 정보 보존
                shape_dict = {
                    'label': label,
                    'shape': shape,
                    'original_group_id': original_group_id,
                    'object_idx': obj_idx,
                    'part_idx': part_idx,
                    'total_parts': len(shapes)
                }
                
                exploded_objects.append((
                    annotator_idx,
                    obj_idx,
                    part_idx,
                    shape_dict,
                    poly,
                    original_group_id
                ))
            except Exception as e:
                print(f"Warning: Failed to convert shape to polygon: {e}")
                continue
    
    return exploded_objects


def calculate_iou(poly1, poly2) -> float:
    """Calculate IoU between two geometries."""
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    
    try:
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        
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
    Find clusters of objects using IoU-based graph connectivity.
    
    Args:
        all_objects: List of (annotator_idx, object_idx, part_idx, shape_dict, polygon, original_group_id)
        iou_threshold: Minimum IoU to connect two objects
    
    Returns:
        List of sets, each set contains indices into all_objects
    """
    n = len(all_objects)
    
    # Build adjacency list
    adj = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            poly_i = all_objects[i][4]
            poly_j = all_objects[j][4]
            
            if poly_i is not None and poly_j is not None:
                iou = calculate_iou(poly_i, poly_j)
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


def reconstruct_multipolygon_clusters(
    cluster_objects: List[Tuple],
    merge_method: str
) -> Dict:
    """
    Cluster 내 객체들을 원본 group_id 기반으로 재그룹화하여 MultiPolygon을 재구성한다.
    
    Args:
        cluster_objects: List of (annotator_idx, object_idx, part_idx, shape_dict, polygon, original_group_id)
        merge_method: 'union' or 'intersection'
    
    Returns:
        Dict with reconstructed object information
    """
    # 어노테이터별 원본 객체(group_id) 단위로 그룹화
    annotator_objects = defaultdict(lambda: defaultdict(list))
    
    for obj in cluster_objects:
        ann_idx, obj_idx, part_idx, shape_dict, poly, orig_group_id = obj
        
        # 원본 객체의 고유 식별자: (annotator_idx, object_idx)
        original_object_key = (ann_idx, obj_idx)
        annotator_objects[ann_idx][original_object_key].append(obj)
    
    # 각 어노테이터가 이 cluster에 기여한 원본 객체 수
    contributions_per_annotator = {ann_idx: len(objs) for ann_idx, objs in annotator_objects.items()}
    
    # 각 어노테이터는 최대 1개의 원본 객체만 기여해야 함
    if any(count > 1 for count in contributions_per_annotator.values()):
        return {'valid': False, 'reason': 'ambiguous_annotator_contribution'}
    
    # 최소 2명의 어노테이터가 참여해야 함
    if len(annotator_objects) < 2:
        return {'valid': False, 'reason': 'insufficient_annotators'}
    
    # 각 어노테이터의 기여를 MultiPolygon으로 재구성
    annotator_multipolygons = {}
    annotator_metadata = {}
    
    for ann_idx, objects_dict in annotator_objects.items():
        # 각 어노테이터는 1개의 원본 객체만 기여함
        original_object_key = list(objects_dict.keys())[0]
        parts = objects_dict[original_object_key]
        
        # parts를 MultiPolygon으로 결합
        polygons = [part[4] for part in parts]
        
        if len(polygons) == 1:
            multipolygon = polygons[0]
        else:
            multipolygon = MultiPolygon(polygons)
        
        annotator_multipolygons[ann_idx] = multipolygon
        
        # metadata 저장
        shape_dicts = [part[3] for part in parts]
        annotator_metadata[ann_idx] = {
            'label': shape_dicts[0]['label'],
            'description': shape_dicts[0]['shape'].get('description', ''),
            'original_group_id': shape_dicts[0]['original_group_id'],
            'object_idx': shape_dicts[0]['object_idx'],
            'total_parts': len(parts)
        }
    
    # IoU 계산 (MultiPolygon 단위)
    annotator_indices = list(annotator_multipolygons.keys())
    ious = []
    for i in range(len(annotator_indices)):
        for j in range(i + 1, len(annotator_indices)):
            ann_i = annotator_indices[i]
            ann_j = annotator_indices[j]
            iou = calculate_iou(
                annotator_multipolygons[ann_i],
                annotator_multipolygons[ann_j]
            )
            ious.append(iou)
    
    min_iou = min(ious) if ious else 0.0
    
    # Label voting
    labels = [meta['label'] for meta in annotator_metadata.values()]
    label_counts = Counter(labels)
    most_common = label_counts.most_common()
    
    num_annotators = len(annotator_indices)
    
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
    descriptions = [meta['description'] for meta in annotator_metadata.values()]
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
    
    # Merge polygons
    polygons_to_merge = list(annotator_multipolygons.values())
    merged_points_list = merge_polygons(polygons_to_merge, merge_method)
    
    return {
        'valid': True,
        'final_label': final_label,
        'final_description': final_description,
        'vote_status': vote_status,
        'min_iou': min_iou,
        'merged_points_list': merged_points_list,
        'num_annotators': num_annotators,
        'annotator_metadata': annotator_metadata,
        'labels': labels,
        'descriptions': descriptions
    }


def merge_polygons_union(polygons: List) -> List[List[List[float]]]:
    """Merge polygons using union. Returns list of polygon coordinates."""
    flattened = []
    for geom in polygons:
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


def merge_polygons_intersection(polygons: List) -> List[List[List[float]]]:
    """Merge polygons using intersection. Returns list of polygon coordinates."""
    single_geoms = []
    for geom in polygons:
        if isinstance(geom, MultiPolygon):
            single_geoms.append(unary_union(geom))
        else:
            single_geoms.append(geom)
    
    merged = single_geoms[0]
    for poly in single_geoms[1:]:
        merged = merged.intersection(poly)
    
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


def merge_polygons(polygons: List, method: str) -> List[List[List[float]]]:
    """Merge polygons. Returns list of polygon coordinates."""
    if method == 'union':
        return merge_polygons_union(polygons)
    elif method == 'intersection':
        return merge_polygons_intersection(polygons)
    else:
        raise ValueError(f"Unknown merge method: {method}")


def load_annotations_by_image(input_dir: str) -> Dict[str, List[Dict]]:
    """Load all annotations grouped by image name."""
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


def process_image_annotations_clustering(
    image_name: str,
    annotations: List[Dict],
    iou_threshold: float,
    merge_method: str
) -> Tuple[Optional[Dict], Dict]:
    """
    Process annotations using IoU-based clustering with fixed MultiPolygon handling.
    
    Args:
        image_name: Name of the image
        annotations: List of annotation dictionaries (2-3 annotators)
        iou_threshold: IoU threshold for clustering
        merge_method: Method to merge polygons
    
    Returns:
        Tuple of (merged_annotation_dict or None, report_dict)
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
    
    num_total_annotators = len(annotations)
    annotator_names = [ann['annotator'] for ann in annotations]
    report['annotators'] = annotator_names
    
    # Extract and group shapes from each annotator
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
    
    # Explode MultiPolygon objects into individual parts
    all_objects = []
    for ann_idx, grouped_shapes in enumerate(all_grouped_shapes):
        exploded = explode_multipolygon_objects(grouped_shapes, ann_idx)
        all_objects.extend(exploded)
    
    print(f"  Image {image_name}: Exploded {sum(report['num_objects_per_annotator'])} objects into {len(all_objects)} parts")
    
    # Find clusters by IoU (now working with individual parts)
    clusters = find_clusters_by_iou(all_objects, iou_threshold)
    
    # Process each cluster
    merged_shapes = []
    rejected_count = 0
    
    for cluster_idx, cluster_indices in enumerate(clusters):
        cluster_objects = [all_objects[i] for i in cluster_indices]
        
        # Reconstruct multipolygon and validate cluster
        result = reconstruct_multipolygon_clusters(cluster_objects, merge_method)
        
        if not result['valid']:
            rejected_count += 1
            continue
        
        # Create output shapes
        merged_points_list = result['merged_points_list']
        annotator_names_in_cluster = [annotator_names[ann_idx] 
                                       for ann_idx in result['annotator_metadata'].keys()]
        
        # Prepare annotator details
        annotator_details = []
        for ann_idx, meta in result['annotator_metadata'].items():
            annotator_details.append({
                'name': annotator_names[ann_idx],
                'label': meta['label'],
                'description': meta['description'],
                'object_index': meta['object_idx'],
                'total_parts': meta['total_parts'],
                'original_group_id': meta['original_group_id']
            })
        
        # Assign group_id if multiple parts
        assigned_group_id = None
        if len(merged_points_list) > 1:
            assigned_group_id = cluster_idx + 10000
        
        # Create shapes
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
                    'annotators': annotator_names_in_cluster,
                    'annotator_labels': result['labels'],
                    'annotator_descriptions': result['descriptions'],
                    'annotator_details': annotator_details,
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
    
    # Update report
    all_labels = [s['label'] for s in merged_shapes]
    report['final_label'] = ','.join(set(all_labels))
    
    vote_statuses = [s['attributes']['vote_status'] for s in merged_shapes]
    if all(vs == 'unanimous' for vs in vote_statuses):
        report['vote_status'] = 'unanimous'
    elif any(vs == 'voted' for vs in vote_statuses):
        report['vote_status'] = 'voted'
    else:
        report['vote_status'] = 'clean'
    
    # Create merged annotation
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
            'algorithm': 'iou_clustering_v7_multipolygon_fixed'
        }
    }
    
    return merged_annotation, report


def main():
    """Main execution function."""
    args = parse_args()
    
    print("=" * 80)
    print("Annotation Merger v7 (IoU-based Clustering with MultiPolygon Fix)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input directory:  {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  IoU threshold:    {args.iou_threshold}")
    print(f"  Merge method:     {args.merge_method}")
    print(f"\nAlgorithm:")
    print(f"  MultiPolygon parts are exploded into individual polygons for clustering")
    print(f"  After clustering, parts are reconstructed based on original group_id")
    print(f"  This ensures accurate IoU calculation between multipolygon objects")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    annotations_by_image = load_annotations_by_image(args.input_dir)
    
    print("\n" + "=" * 80)
    print("Processing images...")
    print("=" * 80)
    
    merged_annotations = []
    reports = []
    
    for image_name, annotations in annotations_by_image.items():
        merged, report = process_image_annotations_clustering(
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
python merge_annotations_v7_clustering.py \
    --input-dir /Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st_validation \
    --output-dir /Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st_validation_output \
    --iou-threshold 0.5 \
    --merge-method union
"""