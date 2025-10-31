#!/usr/bin/env python3
"""
Merge annotation JSON files from multiple annotators with quality control.

This script merges LabelMe-format annotations from 3 annotators per image,
performing IoU-based quality checks and label voting.
"""

import argparse
import json
import os
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.ops import unary_union
from datetime import datetime


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
    Process annotations for a single image.

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
        'num_annotators': len(annotations)
    }

    # Check if we have exactly 3 annotators
    if len(annotations) != 3:
        report['mask_error'] = True
        report['vote_status'] = f'wrong_count_{len(annotations)}'
        return None, report

    # Extract shapes and labels from each annotator
    all_shapes = []
    all_labels = []

    for ann in annotations:
        shapes = ann['data'].get('shapes', [])
        if not shapes:
            report['mask_error'] = True
            report['vote_status'] = 'no_shapes'
            return None, report

        # For now, assume one shape per image (can be extended)
        all_shapes.append(shapes)
        all_labels.append([s['label'] for s in shapes])

    # Process each shape index (assuming corresponding shapes across annotators)
    # For simplicity, we'll process the first shape from each annotator
    if not all(len(shapes) > 0 for shapes in all_shapes):
        report['mask_error'] = True
        report['vote_status'] = 'missing_shapes'
        return None, report

    # Get the primary shape from each annotator
    primary_shapes = [shapes[0] for shapes in all_shapes]
    primary_labels = [s['label'] for s in primary_shapes]
    report['annotator_labels'] = primary_labels

    # Convert to polygons
    try:
        polygons = [polygon_from_points(s['points']) for s in primary_shapes]
    except Exception as e:
        report['mask_error'] = True
        report['vote_status'] = f'invalid_polygon: {e}'
        return None, report

    # Calculate pairwise IoU
    ious = []
    for i in range(len(polygons)):
        for j in range(i + 1, len(polygons)):
            iou = calculate_iou(polygons[i], polygons[j])
            ious.append(iou)

    min_iou = min(ious) if ious else 0.0
    report['min_iou'] = round(min_iou, 3)

    # Check if all IoUs meet threshold
    if min_iou < iou_threshold:
        report['mask_error'] = True
        report['vote_status'] = f'low_iou_{min_iou:.3f}'
        return None, report

    # Merge polygons
    try:
        merged_points = merge_polygons(polygons, merge_method)
    except Exception as e:
        report['mask_error'] = True
        report['vote_status'] = f'merge_failed: {e}'
        return None, report

    # Label voting
    label_counts = Counter(primary_labels)
    most_common = label_counts.most_common()

    if len(most_common) == 1:
        # All labels are the same
        final_label = most_common[0][0]
        report['vote_status'] = 'unanimous'
    elif most_common[0][1] > most_common[1][1]:
        # Clear majority
        final_label = most_common[0][0]
        report['vote_status'] = 'voted'
    else:
        # No majority (e.g., 3 different labels or tie)
        report['class_error'] = True
        report['vote_status'] = f'no_majority_{dict(label_counts)}'
        return None, report

    report['final_label'] = final_label

    # Create merged annotation in LabelMe format
    base_data = annotations[0]['data'].copy()
    merged_annotation = {
        'version': base_data.get('version', '5.0.0'),
        'flags': {},
        'shapes': [{
            'label': final_label,
            'points': merged_points,
            'group_id': None,
            'shape_type': 'polygon',
            'flags': {},
            'description': f'Merged from {len(annotations)} annotators using {merge_method}',
            'score': None,
            'difficult': False,
            'attributes': {},
            'kie_linking': []
        }],
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
