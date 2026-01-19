#!/usr/bin/env python3
# merge_annotations_v6_clustering.py

"""
Merge annotation JSON files using IoU-based clustering (instead of 1:1:1 matching).

MAJOR CHANGE from v3:
- v3: Object count must match across annotators (e.g., [5,5,5] or [5,5,3]→use 2 with 5)
- v4: Flexible clustering - each object voted on independently by 2-3 annotators
  * Example: [5,5,3] → may result in 6 objects if clusters form properly
  * Object A: voted by annotator 1,2 → include
  * Object B: voted by annotator 1,3 → include
  * Object C: voted by annotator 2,3 → include

=============================================================================
CLUSTERING ALGORITHM
=============================================================================

1. Pool all objects from all annotators (2-3 annotators per image)
2. Compute pairwise IoU between all objects
3. Build graph: nodes = objects, edges = IoU ≥ threshold
4. Find connected components (clusters)
5. For each cluster:
   - Check: each annotator contributes ≤ 1 object (reject if ambiguous)
   - Check: ≥ 2 annotators in cluster (reject if only 1)
   - Vote on label (2 annotators: unanimous, 3 annotators: majority)
   - Vote on description attributes (comma-separated, each voted independently)
   - Merge polygons using specified method
6. Output all valid clusters as merged objects

=============================================================================
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
        description='Merge annotations using IoU-based clustering'
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


def shapes_to_polygon(grouped_shape: Dict):
    """Convert grouped shape(s) to Shapely Polygon or MultiPolygon."""
    shapes = grouped_shape['shapes']
    
    if len(shapes) == 1:
        return polygon_from_points(shapes[0]['points'])
    else:
        polygons = [polygon_from_points(shape['points']) for shape in shapes]
        return MultiPolygon(polygons)


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


def merge_polygons_union(polygons: List) -> List[List[float]]:
    """Merge polygons using union."""
    flattened = []
    for geom in polygons:
        if isinstance(geom, MultiPolygon):
            flattened.extend(geom.geoms)
        else:
            flattened.append(geom)
    
    merged = unary_union(flattened)
    
    if merged.geom_type == 'MultiPolygon':
        merged = max(merged.geoms, key=lambda p: p.area)
    
    coords = list(merged.exterior.coords[:-1])
    return [[float(x), float(y)] for x, y in coords]


def merge_polygons_intersection(polygons: List) -> List[List[float]]:
    """Merge polygons using intersection."""
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
    elif merged.geom_type == 'MultiPolygon':
        merged = max(merged.geoms, key=lambda p: p.area)
    
    coords = list(merged.exterior.coords[:-1])
    return [[float(x), float(y)] for x, y in coords]


def merge_polygons(polygons: List, method: str) -> List[List[float]]:
    """Merge multiple polygons using specified method."""
    if method == 'union':
        return merge_polygons_union(polygons)
    elif method == 'intersection':
        return merge_polygons_intersection(polygons)
    else:
        raise ValueError(f"Unknown merge method: {method}")


def find_clusters_by_iou(
    all_objects: List[Tuple[int, int, Dict, any]],
    iou_threshold: float
) -> List[Set[int]]:
    """
    Find clusters of objects using IoU-based graph connectivity.
    
    Args:
        all_objects: List of (annotator_idx, object_idx, grouped_shape, polygon)
        iou_threshold: Minimum IoU to connect two objects
    
    Returns:
        List of sets, each set contains indices into all_objects
    """
    n = len(all_objects)
    
    # Build adjacency list
    adj = defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            poly_i = all_objects[i][3]
            poly_j = all_objects[j][3]
            
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


# Part 2 of merge_annotations_v4_clustering.py

def process_image_annotations_clustering(
    image_name: str,
    annotations: List[Dict],
    iou_threshold: float,
    merge_method: str
) -> Tuple[Optional[Dict], Dict]:
    """
    Process annotations using IoU-based clustering (v4 algorithm).
    
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
    
    # Check if we have 2 or 3 annotators
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
    
    # Check if any annotator has no shapes
    if any(len(g) == 0 for g in all_grouped_shapes):
        report['mask_error'] = True
        report['vote_status'] = 'no_shapes'
        return None, report
    
    # Build pool of all objects: (annotator_idx, object_idx, grouped_shape, polygon)
    all_objects = []
    for ann_idx, grouped_shapes in enumerate(all_grouped_shapes):
        for obj_idx, grouped_shape in enumerate(grouped_shapes):
            try:
                poly = shapes_to_polygon(grouped_shape)
                all_objects.append((ann_idx, obj_idx, grouped_shape, poly))
            except Exception as e:
                print(f"Warning: Failed to convert object to polygon: {e}")
                all_objects.append((ann_idx, obj_idx, grouped_shape, None))
    
    # Find clusters by IoU
    clusters = find_clusters_by_iou(all_objects, iou_threshold)
    
    # Process each cluster
    merged_shapes = []
    rejected_count = 0
    
    for cluster_idx, cluster_indices in enumerate(clusters):
        # Get objects in this cluster
        cluster_objects = [all_objects[i] for i in cluster_indices]
        
        # Check: each annotator should contribute at most 1 object
        annotators_in_cluster = [obj[0] for obj in cluster_objects]
        if len(annotators_in_cluster) != len(set(annotators_in_cluster)):
            # Ambiguous: same annotator has multiple objects in cluster
            rejected_count += 1
            continue
        
        # Check: at least 2 annotators must participate
        if len(annotators_in_cluster) < 2:
            # Only 1 annotator marked this object
            rejected_count += 1
            continue
        
        num_annotators_in_cluster = len(annotators_in_cluster)
        
        # Extract data for voting
        labels = [obj[2]['label'] for obj in cluster_objects]
        polygons = [obj[3] for obj in cluster_objects]
        grouped_shapes = [obj[2] for obj in cluster_objects]
        
        # Check all polygons are valid
        if any(p is None for p in polygons):
            rejected_count += 1
            continue
        
        # Calculate pairwise IoU (quality check)
        ious = []
        for i in range(len(polygons)):
            for j in range(i + 1, len(polygons)):
                iou = calculate_iou(polygons[i], polygons[j])
                ious.append(iou)
        
        min_iou = min(ious) if ious else 0.0
        
        # Check if all IoUs meet threshold
        if min_iou < iou_threshold:
            rejected_count += 1
            continue
        
        # Label voting
        label_counts = Counter(labels)
        most_common = label_counts.most_common()
        
        if num_annotators_in_cluster == 2:
            # 2 annotators: must be unanimous
            if len(most_common) == 1:
                final_label = most_common[0][0]
                vote_status = 'unanimous'
            else:
                rejected_count += 1
                continue
        else:  # num_annotators_in_cluster == 3
            # 3 annotators: unanimous or majority
            if len(most_common) == 1:
                final_label = most_common[0][0]
                vote_status = 'unanimous'
            elif most_common[0][1] > most_common[1][1]:
                final_label = most_common[0][0]
                vote_status = 'voted'
            else:
                rejected_count += 1
                continue
        
        # Description voting (attribute-wise)
        all_attributes = []
        original_descriptions = []
        for grouped in grouped_shapes:
            if grouped['shapes']:
                desc = grouped['shapes'][0].get('description', '').strip()
                original_descriptions.append(desc)
                if desc:
                    attrs = {attr.strip() for attr in desc.split(',') if attr.strip()}
                    all_attributes.append(attrs)
                else:
                    all_attributes.append(set())
            else:
                original_descriptions.append('')
                all_attributes.append(set())
        
        unique_attrs = set()
        for attrs in all_attributes:
            unique_attrs.update(attrs)
        
        final_attrs = []
        if num_annotators_in_cluster == 2:
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
        try:
            merged_points = merge_polygons(polygons, merge_method)
        except Exception as e:
            rejected_count += 1
            continue
        
        # Build annotator details
        annotator_details = []
        cluster_annotator_names = []
        for obj in cluster_objects:
            ann_idx, obj_idx, grouped, poly = obj
            ann_name = annotator_names[ann_idx]
            cluster_annotator_names.append(ann_name)
            
            desc = ''
            if grouped['shapes']:
                desc = grouped['shapes'][0].get('description', '')
            
            annotator_details.append({
                'name': ann_name,
                'label': grouped['label'],
                'description': desc,
                'object_index': obj_idx
            })
        
        # Add merged shape
        group_ids = [obj[2]['group_id'] for obj in cluster_objects]
        group_id_info = ""
        if any(gid is not None for gid in group_ids):
            group_id_info = f" (grouped shapes: {group_ids})"
        
        merged_shapes.append({
            'label': final_label,
            'points': merged_points,
            'group_id': None,
            'shape_type': 'polygon',
            'flags': {},
            'description': final_description,
            'score': None,
            'difficult': False,
            'attributes': {
                'annotators': cluster_annotator_names,
                'annotator_labels': labels,
                'annotator_descriptions': original_descriptions,
                'annotator_details': annotator_details,
                'min_iou': float(round(min_iou, 3)),
                'vote_status': vote_status,
                'merge_info': f'Cluster {cluster_idx+1}: {num_annotators_in_cluster} annotators, merged using {merge_method}{group_id_info}'
            },
            'kie_linking': []
        })
    
    report['num_clusters'] = len(merged_shapes)
    report['rejected_clusters'] = rejected_count
    
    if len(merged_shapes) == 0:
        report['mask_error'] = True
        report['vote_status'] = 'no_valid_clusters'
        return None, report
    
    # Update report
    all_labels = [s['label'] for s in merged_shapes]
    report['final_label'] = ','.join(all_labels)
    
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
            'totalClusters': int(len(merged_shapes)),
            'rejectedClusters': int(rejected_count),
            'objectCountsPerAnnotator': report['num_objects_per_annotator'],
            'algorithm': 'iou_clustering_v4'
        }
    }
    
    return merged_annotation, report


def main():
    """Main execution function."""
    args = parse_args()
    
    print("=" * 80)
    print("Annotation Merger v4 (IoU-based Clustering)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input directory:  {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  IoU threshold:    {args.iou_threshold}")
    print(f"  Merge method:     {args.merge_method}")
    print(f"\nAlgorithm:")
    print(f"  Flexible clustering - each object voted on independently")
    print(f"  Objects from different counts can coexist")
    print(f"  Example: [5,5,3] may produce 6 objects if clusters form properly")
    
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
python merge_annotations_v6_clustering.py \
    --input-dir /Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st_validation \
    --output-dir /Users/woojin/Documents/AioT/test/sample_images/Imgenet-ESM-SS/batch_1st/batch_1st_validation_output \
    --iou-threshold 0.5 \
    --merge-method union
"""