#!/usr/bin/env python3
"""
Merge annotation JSON files from multiple annotators with quality control.

This script merges LabelMe-format annotations from 3 annotators per image,
performing IoU-based quality checks and label voting.
"""

import argparse
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union
from datetime import datetime
from scipy.optimize import linear_sum_assignment


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Merge annotation JSON files from multiple annotators'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='./annotations/',
        help='Input directory containing annotator subdirectories (default: ./annotations/)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./merged_results/',
        help='Output directory for merged JSON and CSV (default: ./merged_results/)'
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for mask agreement (default: 0.5)'
    )
    parser.add_argument(
        '--merge-method',
        type=str,
        choices=['union', 'average', 'intersection'],
        default='union',
        help='Method to merge polygons: union (largest), average (middle), intersection (conservative)'
    )
    return parser.parse_args()


def polygon_from_points(points: List[List[float]]) -> Polygon:
    """
    Convert list of points to Shapely Polygon.

    Args:
        points: List of [x, y] coordinates

    Returns:
        Shapely Polygon object
    """
    if len(points) < 3:
        raise ValueError(f"Polygon must have at least 3 points, got {len(points)}")
    return Polygon(points)


def calculate_iou(poly1: Polygon, poly2: Polygon) -> float:
    """
    Calculate Intersection over Union (IoU) between two polygons.

    Args:
        poly1: First polygon
        poly2: Second polygon

    Returns:
        IoU value between 0 and 1
    """
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


def merge_polygons_union(polygons: List[Polygon]) -> List[List[float]]:
    """
    Merge polygons using union (largest coverage).

    Args:
        polygons: List of Shapely Polygon objects

    Returns:
        List of [x, y] coordinates for merged polygon
    """
    merged = unary_union(polygons)

    # Handle case where union creates MultiPolygon - take the largest
    if merged.geom_type == 'MultiPolygon':
        merged = max(merged.geoms, key=lambda p: p.area)

    # Extract exterior coordinates
    coords = list(merged.exterior.coords[:-1])  # Remove duplicate last point
    return [[float(x), float(y)] for x, y in coords]


def merge_polygons_intersection(polygons: List[Polygon]) -> List[List[float]]:
    """
    Merge polygons using intersection (most conservative).

    Args:
        polygons: List of Shapely Polygon objects

    Returns:
        List of [x, y] coordinates for merged polygon
    """
    merged = polygons[0]
    for poly in polygons[1:]:
        merged = merged.intersection(poly)

    # Handle case where intersection is empty or creates GeometryCollection
    if merged.is_empty or merged.geom_type not in ['Polygon', 'MultiPolygon']:
        # Fallback to first polygon if intersection fails
        merged = polygons[0]
    elif merged.geom_type == 'MultiPolygon':
        merged = max(merged.geoms, key=lambda p: p.area)

    coords = list(merged.exterior.coords[:-1])
    return [[float(x), float(y)] for x, y in coords]


def merge_polygons_average(polygons: List[Polygon]) -> List[List[float]]:
    """
    Merge polygons by averaging coordinates.
    Note: This assumes polygons have similar number of points and shape.

    Args:
        polygons: List of Shapely Polygon objects

    Returns:
        List of [x, y] coordinates for merged polygon
    """
    # Get all coordinate arrays
    all_coords = [np.array(poly.exterior.coords[:-1]) for poly in polygons]

    # Find the polygon with median number of points
    num_points = [len(coords) for coords in all_coords]
    target_num_points = int(np.median(num_points))

    # Resample all polygons to have the same number of points
    resampled_coords = []
    for coords in all_coords:
        if len(coords) != target_num_points:
            # Simple linear interpolation to get target number of points
            indices = np.linspace(0, len(coords) - 1, target_num_points)
            indices_int = indices.astype(int)
            resampled = coords[indices_int]
        else:
            resampled = coords
        resampled_coords.append(resampled)

    # Average the coordinates
    avg_coords = np.mean(resampled_coords, axis=0)

    return [[float(x), float(y)] for x, y in avg_coords]


def merge_polygons(polygons: List[Polygon], method: str) -> List[List[float]]:
    """
    Merge multiple polygons using specified method.

    Args:
        polygons: List of Shapely Polygon objects
        method: 'union', 'average', or 'intersection'

    Returns:
        List of [x, y] coordinates for merged polygon
    """
    if method == 'union':
        return merge_polygons_union(polygons)
    elif method == 'intersection':
        return merge_polygons_intersection(polygons)
    elif method == 'average':
        return merge_polygons_average(polygons)
    else:
        raise ValueError(f"Unknown merge method: {method}")


def match_shapes_across_annotators(
    shapes_lists: List[List[Dict]]
) -> List[Tuple[int, int, int]]:
    """
    Match shapes across 3 annotators using IoU-based Hungarian algorithm.

    Args:
        shapes_lists: List of 3 shape lists, one from each annotator

    Returns:
        List of tuples (idx0, idx1, idx2) representing matched shape indices
    """
    if len(shapes_lists) != 3:
        return []

    # Convert all shapes to polygons
    all_polygons = []
    for shapes in shapes_lists:
        polygons = []
        for shape in shapes:
            try:
                poly = polygon_from_points(shape['points'])
                polygons.append(poly)
            except:
                polygons.append(None)
        all_polygons.append(polygons)

    n0, n1, n2 = len(all_polygons[0]), len(all_polygons[1]), len(all_polygons[2])

    # Step 1: Match annotator 0 with annotator 1
    if n0 == 0 or n1 == 0:
        return []

    cost_matrix_01 = np.zeros((n0, n1))
    for i in range(n0):
        for j in range(n1):
            if all_polygons[0][i] is not None and all_polygons[1][j] is not None:
                iou = calculate_iou(all_polygons[0][i], all_polygons[1][j])
                cost_matrix_01[i, j] = -iou  # Negative because we want to maximize IoU
            else:
                cost_matrix_01[i, j] = 1.0  # High cost for invalid polygons

    row_ind_01, col_ind_01 = linear_sum_assignment(cost_matrix_01)

    # Step 2: For each match from step 1, find best match in annotator 2
    matches = []
    for i, j in zip(row_ind_01, col_ind_01):
        if n2 == 0:
            continue

        # Find best match in annotator 2
        best_k = -1
        best_iou = -1

        for k in range(n2):
            if all_polygons[2][k] is None:
                continue

            # Calculate average IoU with both matched polygons
            iou_0k = calculate_iou(all_polygons[0][i], all_polygons[2][k]) if all_polygons[0][i] else 0
            iou_1k = calculate_iou(all_polygons[1][j], all_polygons[2][k]) if all_polygons[1][j] else 0
            avg_iou = (iou_0k + iou_1k) / 2

            if avg_iou > best_iou:
                best_iou = avg_iou
                best_k = k

        if best_k >= 0:
            matches.append((i, j, best_k))

    return matches


def load_annotations_by_image(input_dir: str) -> Dict[str, List[Dict]]:
    """
    Load all annotations grouped by image name.

    Args:
        input_dir: Directory containing annotator subdirectories

    Returns:
        Dictionary mapping image_name -> list of annotation dicts
    """
    input_path = Path(input_dir)
    annotations_by_image = defaultdict(list)

    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Find all annotator directories
    annotator_dirs = [d for d in input_path.iterdir() if d.is_dir()]

    if not annotator_dirs:
        print(f"Error: No annotator subdirectories found in {input_dir}")
        sys.exit(1)

    print(f"\nFound {len(annotator_dirs)} annotator directories:")
    for d in annotator_dirs:
        print(f"  - {d.name}")

    # Load all JSON files
    total_files = 0
    for annotator_dir in annotator_dirs:
        json_files = list(annotator_dir.glob('*.json'))
        total_files += len(json_files)

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                image_name = data.get('imagePath', json_file.stem)
                annotations_by_image[image_name].append({
                    'annotator': annotator_dir.name,
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
    Process annotations for a single image with multiple objects.

    Args:
        image_name: Name of the image
        annotations: List of annotation dictionaries
        iou_threshold: IoU threshold for quality check
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
        'annotator_labels': [],
        'min_iou': None,
        'num_annotators': len(annotations),
        'num_objects': 0
    }

    # Check if we have exactly 3 annotators
    if len(annotations) != 3:
        report['mask_error'] = True
        report['vote_status'] = f'wrong_count_{len(annotations)}'
        return None, report

    # Extract shapes from each annotator
    all_shapes = []
    for ann in annotations:
        shapes = ann['data'].get('shapes', [])
        all_shapes.append(shapes)

    # Check if all annotators have shapes
    shape_counts = [len(shapes) for shapes in all_shapes]
    if any(count == 0 for count in shape_counts):
        report['mask_error'] = True
        report['vote_status'] = 'no_shapes'
        return None, report

    # Check if all annotators have the same number of objects
    if len(set(shape_counts)) != 1:
        report['mask_error'] = True
        report['vote_status'] = f'count_mismatch_{shape_counts}'
        return None, report

    num_objects = shape_counts[0]
    report['num_objects'] = num_objects

    # Match shapes across annotators using IoU
    matches = match_shapes_across_annotators(all_shapes)

    # Check if we got the expected number of matches
    if len(matches) != num_objects:
        report['mask_error'] = True
        report['vote_status'] = f'matching_failed_{len(matches)}/{num_objects}'
        return None, report

    # Process each matched object
    merged_shapes = []
    all_min_ious = []
    all_vote_statuses = []
    all_labels = []

    for match_idx, (idx0, idx1, idx2) in enumerate(matches):
        # Get the three matched shapes
        shape0 = all_shapes[0][idx0]
        shape1 = all_shapes[1][idx1]
        shape2 = all_shapes[2][idx2]

        matched_shapes = [shape0, shape1, shape2]
        labels = [s['label'] for s in matched_shapes]

        # Convert to polygons
        try:
            polygons = [polygon_from_points(s['points']) for s in matched_shapes]
        except Exception as e:
            report['mask_error'] = True
            report['vote_status'] = f'invalid_polygon_obj{match_idx}: {e}'
            return None, report

        # Calculate pairwise IoU
        ious = []
        for i in range(len(polygons)):
            for j in range(i + 1, len(polygons)):
                iou = calculate_iou(polygons[i], polygons[j])
                ious.append(iou)

        min_iou = min(ious) if ious else 0.0
        all_min_ious.append(min_iou)

        # Check if all IoUs meet threshold
        if min_iou < iou_threshold:
            report['mask_error'] = True
            report['vote_status'] = f'low_iou_obj{match_idx}_{min_iou:.3f}'
            return None, report

        # Merge polygons
        try:
            merged_points = merge_polygons(polygons, merge_method)
        except Exception as e:
            report['mask_error'] = True
            report['vote_status'] = f'merge_failed_obj{match_idx}: {e}'
            return None, report

        # Label voting
        label_counts = Counter(labels)
        most_common = label_counts.most_common()

        if len(most_common) == 1:
            # All labels are the same
            final_label = most_common[0][0]
            vote_status = 'unanimous'
        elif most_common[0][1] > most_common[1][1]:
            # Clear majority
            final_label = most_common[0][0]
            vote_status = 'voted'
        else:
            # No majority
            report['class_error'] = True
            report['vote_status'] = f'no_majority_obj{match_idx}_{dict(label_counts)}'
            return None, report

        all_vote_statuses.append(vote_status)
        all_labels.append(final_label)

        # Add merged shape
        merged_shapes.append({
            'label': final_label,
            'points': merged_points,
            'group_id': None,
            'shape_type': 'polygon',
            'flags': {},
            'description': f'Merged obj {match_idx+1} from 3 annotators using {merge_method}',
            'score': None,
            'difficult': False,
            'attributes': {},
            'kie_linking': []
        })

    # Update report with overall statistics
    report['min_iou'] = round(min(all_min_ious), 3) if all_min_ious else None
    report['final_label'] = ','.join(all_labels) if all_labels else None
    report['annotator_labels'] = all_labels

    # Overall vote status
    if all(vs == 'unanimous' for vs in all_vote_statuses):
        report['vote_status'] = 'unanimous'
    elif any(vs == 'voted' for vs in all_vote_statuses):
        report['vote_status'] = 'voted'
    else:
        report['vote_status'] = 'clean'

    # Create merged annotation in LabelMe format
    base_data = annotations[0]['data'].copy()
    merged_annotation = {
        'version': base_data.get('version', '5.0.0'),
        'flags': {},
        'shapes': merged_shapes,
        'imagePath': image_name,
        'imageData': None,
        'imageHeight': base_data.get('imageHeight'),
        'imageWidth': base_data.get('imageWidth')
    }

    return merged_annotation, report


def main():
    """Main execution function."""
    args = parse_args()

    print("=" * 80)
    print("Annotation Merger")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Input directory:  {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  IoU threshold:    {args.iou_threshold}")
    print(f"  Merge method:     {args.merge_method}")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load all annotations
    annotations_by_image = load_annotations_by_image(args.input_dir)

    # Process each image
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

        # Progress indicator
        status = "✓" if merged is not None else "✗"
        print(f"  {status} {image_name}: {report['vote_status']}")

    # Save merged annotations
    output_json = output_path / 'merged_annotations.json'

    # For LabelMe format, we need to save each image as a separate file
    # Or we can save all in a list
    # Let's create a directory for individual files
    merged_dir = output_path / 'merged_annotations'
    merged_dir.mkdir(exist_ok=True)

    for merged in merged_annotations:
        image_name = merged['imagePath']
        # Remove extension and add .json
        base_name = Path(image_name).stem + '.json'
        output_file = merged_dir / base_name

        with open(output_file, 'w') as f:
            json.dump(merged, f, indent=2)

    print(f"\n✓ Saved {len(merged_annotations)} merged annotations to {merged_dir}/")

    # Save report CSV
    df = pd.DataFrame(reports)
    csv_path = output_path / 'merge_report.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved report to {csv_path}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"  Total images:           {len(reports)}")
    print(f"  Successfully merged:    {len(merged_annotations)}")
    print(f"  Mask errors:            {sum(r['mask_error'] for r in reports)}")
    print(f"  Class errors:           {sum(r['class_error'] for r in reports)}")
    print(f"  Voted (not unanimous):  {sum(r['vote_status'] == 'voted' for r in reports)}")
    print(f"  Unanimous:              {sum(r['vote_status'] == 'unanimous' for r in reports)}")

    # Show IoU statistics for successful merges
    successful_ious = [r['min_iou'] for r in reports if r['min_iou'] is not None and not r['mask_error']]
    if successful_ious:
        print(f"\n  IoU Statistics (successful merges):")
        print(f"    Mean: {np.mean(successful_ious):.3f}")
        print(f"    Min:  {np.min(successful_ious):.3f}")
        print(f"    Max:  {np.max(successful_ious):.3f}")

    print("\n" + "=" * 80)
    print(f"Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
