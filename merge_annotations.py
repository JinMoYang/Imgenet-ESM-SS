#!/usr/bin/env python3
"""
Merge annotation JSON files from multiple annotators with quality control.

This script merges LabelMe-format annotations from 3 annotators per image,
performing IoU-based quality checks and label voting.

=============================================================================
DIRECTORY STRUCTURE REQUIREMENTS
=============================================================================

Expected input directory structure:
    annotations/
    ├── annotator1/
    │   ├── image1.json
    │   ├── image2.json
    │   └── ...
    ├── annotator2/
    │   ├── image1.json
    │   ├── image2.json
    │   └── ...
    └── annotator3/
        ├── image1.json
        ├── image2.json
        └── ...

Output directory structure (created automatically):
    merged_results/
    ├── merged_annotations/      # Individual merged JSON files
    │   ├── image1.json
    │   ├── image2.json
    │   └── ...
    └── merge_report.csv         # Quality control report

=============================================================================
GROUP_ID SUPPORT
=============================================================================

This script supports multi-part annotations using LabelMe's group_id field.
If an object consists of multiple disconnected regions (e.g., a lemon with
an occluded part), annotators can assign the same group_id to all parts,
and they will be treated as a single object during matching and IoU calculation.

Example use case:
- Object 1: Main body of lemon (group_id: 1)
- Object 2: Occluded part of lemon (group_id: 1)
- Object 3: Another lemon (group_id: 2)

The script will:
1. Group shapes with same group_id into one object (1 & 2 → one object)
2. Convert to MultiPolygon for IoU calculation
3. Match across annotators based on combined IoU
4. Merge all parts into a single polygon in output

=============================================================================
QUALITY CONTROL PROCESS
=============================================================================

For each image:
1. Check that exactly 3 annotators provided annotations
2. Group shapes by group_id (shapes with same group_id = 1 object)
3. Verify all annotators marked the same number of objects
4. Match objects across annotators using IoU-based Hungarian algorithm
5. For each matched object:
   - Calculate pairwise IoU between all 3 annotations
   - Reject if minimum IoU < threshold (default: 0.5)
   - Vote on label (unanimous or majority required)
   - Merge polygons using specified method (union/intersection)
6. Output merged annotation only if all checks pass

=============================================================================
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
        choices=['union', 'intersection'],
        default='union',
        help='Method to merge polygons: union (largest) or intersection (conservative)'
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


def group_shapes_by_group_id(shapes: List[Dict]) -> List[Dict]:
    """
    Group shapes by their group_id. Shapes with the same group_id are treated as one object.

    This is the KEY function that enables multi-part annotation support.

    IMPORTANT: This function changes the unit of comparison from individual shapes
    to logical objects. Multiple disconnected shapes with the same group_id become
    ONE object for the purposes of:
    - Object counting (3 shapes with group_id=1 count as 1 object)
    - IoU calculation (all parts combined)
    - Label voting (one vote per object, not per shape)

    Example:
        Input shapes:
        [
            {'label': 'lemon', 'group_id': 1, 'points': [...]},  # Main body
            {'label': 'lemon', 'group_id': 1, 'points': [...]},  # Occluded part
            {'label': 'apple', 'group_id': None, 'points': [...]}  # Separate object
        ]

        Output grouped shapes:
        [
            {
                'shapes': [shape1, shape2],  # Both parts of the lemon
                'group_id': 1,
                'label': 'lemon'
            },
            {
                'shapes': [shape3],  # The apple
                'group_id': None,
                'label': 'apple'
            }
        ]

    Args:
        shapes: List of shape dictionaries from LabelMe format, where each shape may have:
                - 'label': object class name
                - 'group_id': integer or None (None means ungrouped)
                - 'points': list of [x, y] coordinates
                - other LabelMe fields...

    Returns:
        List of grouped shape dictionaries, where each dict represents ONE LOGICAL OBJECT
        with fields:
        - 'shapes': list of one or more shape dicts (multiple if grouped)
        - 'group_id': the shared group_id or None
        - 'label': the object label (from first shape if grouped)
    """
    from collections import OrderedDict

    # Use OrderedDict to preserve annotation order (important for consistent matching)
    grouped = OrderedDict()
    ungrouped = []

    for shape in shapes:
        group_id = shape.get('group_id')

        if group_id is None:
            # No group_id - treat as individual object
            # Each ungrouped shape becomes its own object
            ungrouped.append({
                'shapes': [shape],
                'group_id': None,
                'label': shape['label']
            })
        else:
            # Has group_id - add to existing group or create new one
            if group_id not in grouped:
                grouped[group_id] = {
                    'shapes': [],
                    'group_id': group_id,
                    'label': shape['label']  # Take label from first shape in group
                }
            grouped[group_id]['shapes'].append(shape)

    # Combine grouped and ungrouped objects
    # Order: grouped objects first (by first appearance), then ungrouped
    result = list(grouped.values()) + ungrouped
    return result


def shapes_to_polygon(grouped_shape: Dict):
    """
    Convert grouped shape(s) to Shapely Polygon or MultiPolygon.

    This function bridges the gap between LabelMe's annotation format and Shapely's
    geometry objects, handling both single-part and multi-part objects.

    KEY BEHAVIOR:
    - If grouped_shape has 1 shape: Returns Polygon (single region)
    - If grouped_shape has 2+ shapes: Returns MultiPolygon (disconnected regions)

    The resulting geometry is used for:
    1. IoU calculation between annotators
    2. Hungarian algorithm matching
    3. Polygon merging

    Example 1 - Single part object:
        grouped_shape = {
            'shapes': [{'points': [[0,0], [1,0], [1,1], [0,1]]}],
            'group_id': None,
            'label': 'apple'
        }
        → Returns: Polygon with 4 vertices

    Example 2 - Multi-part object (same group_id):
        grouped_shape = {
            'shapes': [
                {'points': [[0,0], [1,0], [1,1], [0,1]]},  # Main body
                {'points': [[2,2], [3,2], [3,3], [2,3]]}   # Occluded part
            ],
            'group_id': 1,
            'label': 'lemon'
        }
        → Returns: MultiPolygon with 2 separate polygons

    Args:
        grouped_shape: Dict with fields:
                      - 'shapes': list of shape dicts (from group_shapes_by_group_id)
                      - 'group_id': integer or None
                      - 'label': object label

    Returns:
        Shapely Polygon (if single shape) or MultiPolygon (if multiple shapes)

    Raises:
        ValueError: If any shape has < 3 points (via polygon_from_points)
    """
    from shapely.geometry import MultiPolygon

    shapes = grouped_shape['shapes']

    if len(shapes) == 1:
        # Single shape - return simple Polygon
        # This is the most common case (ungrouped annotations)
        return polygon_from_points(shapes[0]['points'])
    else:
        # Multiple shapes with same group_id - return MultiPolygon
        # Each part becomes a separate polygon within the MultiPolygon
        # IoU calculation will consider the union of all parts
        polygons = [polygon_from_points(shape['points']) for shape in shapes]
        return MultiPolygon(polygons)


def calculate_iou(poly1, poly2) -> float:
    """
    Calculate Intersection over Union (IoU) between two geometries.

    Works with Polygon, MultiPolygon, or any Shapely geometry.

    Args:
        poly1: First geometry (Polygon or MultiPolygon)
        poly2: Second geometry (Polygon or MultiPolygon)

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


def merge_polygons_union(polygons: List) -> List[List[float]]:
    """
    Merge polygons using union (largest coverage).

    Args:
        polygons: List of Shapely Polygon or MultiPolygon objects

    Returns:
        List of [x, y] coordinates for merged polygon
    """
    from shapely.geometry import MultiPolygon

    # Flatten any MultiPolygons into individual polygons
    flattened = []
    for geom in polygons:
        if isinstance(geom, MultiPolygon):
            flattened.extend(geom.geoms)
        else:
            flattened.append(geom)

    merged = unary_union(flattened)

    # Handle case where union creates MultiPolygon - take the largest
    if merged.geom_type == 'MultiPolygon':
        merged = max(merged.geoms, key=lambda p: p.area)

    # Extract exterior coordinates
    coords = list(merged.exterior.coords[:-1])  # Remove duplicate last point
    return [[float(x), float(y)] for x, y in coords]


def merge_polygons_intersection(polygons: List) -> List[List[float]]:
    """
    Merge polygons using intersection (most conservative).

    Args:
        polygons: List of Shapely Polygon or MultiPolygon objects

    Returns:
        List of [x, y] coordinates for merged polygon
    """
    from shapely.geometry import MultiPolygon

    # Flatten any MultiPolygons into individual polygons, then take union of each group
    # to get single geometry per annotator
    single_geoms = []
    for geom in polygons:
        if isinstance(geom, MultiPolygon):
            single_geoms.append(unary_union(geom))
        else:
            single_geoms.append(geom)

    merged = single_geoms[0]
    for poly in single_geoms[1:]:
        merged = merged.intersection(poly)

    # Handle case where intersection is empty or creates GeometryCollection
    if merged.is_empty or merged.geom_type not in ['Polygon', 'MultiPolygon']:
        # Fallback to first polygon if intersection fails
        merged = single_geoms[0]
    elif merged.geom_type == 'MultiPolygon':
        merged = max(merged.geoms, key=lambda p: p.area)

    coords = list(merged.exterior.coords[:-1])
    return [[float(x), float(y)] for x, y in coords]

def merge_polygons(polygons: List, method: str) -> List[List[float]]:
    """
    Merge multiple polygons using specified method.

    Args:
        polygons: List of Shapely Polygon or MultiPolygon objects
        method: 'union', or 'intersection'

    Returns:
        List of [x, y] coordinates for merged polygon
    """
    if method == 'union':
        return merge_polygons_union(polygons)
    elif method == 'intersection':
        return merge_polygons_intersection(polygons)
    else:
        raise ValueError(f"Unknown merge method: {method}")


def match_shapes_across_annotators(
    grouped_shapes_lists: List[List[Dict]]
) -> List[Tuple[int, int, int]]:
    """
    Match shapes across 3 annotators using IoU-based Hungarian algorithm.

    Args:
        grouped_shapes_lists: List of 3 grouped shape lists (output from group_shapes_by_group_id)

    Returns:
        List of tuples (idx0, idx1, idx2) representing matched grouped shape indices
    """
    if len(grouped_shapes_lists) != 3:
        return []

    # Convert all grouped shapes to polygons/multipolygons
    all_polygons = []
    for grouped_shapes in grouped_shapes_lists:
        polygons = []
        for grouped_shape in grouped_shapes:
            try:
                poly = shapes_to_polygon(grouped_shape)
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

    # Extract shapes from each annotator and group by group_id
    all_grouped_shapes = []
    for ann in annotations:
        shapes = ann['data'].get('shapes', [])
        grouped = group_shapes_by_group_id(shapes)
        all_grouped_shapes.append(grouped)

    # Check if all annotators have shapes
    object_counts = [len(grouped) for grouped in all_grouped_shapes]
    if any(count == 0 for count in object_counts):
        report['mask_error'] = True
        report['vote_status'] = 'no_shapes'
        return None, report

    # Check if all annotators have the same number of objects
    if len(set(object_counts)) != 1:
        report['mask_error'] = True
        report['vote_status'] = f'count_mismatch_{object_counts}'
        return None, report

    num_objects = object_counts[0]
    report['num_objects'] = num_objects

    # Match shapes across annotators using IoU
    matches = match_shapes_across_annotators(all_grouped_shapes)

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
        # Get the three matched grouped shapes
        grouped0 = all_grouped_shapes[0][idx0]
        grouped1 = all_grouped_shapes[1][idx1]
        grouped2 = all_grouped_shapes[2][idx2]

        matched_grouped = [grouped0, grouped1, grouped2]
        labels = [g['label'] for g in matched_grouped]
        group_ids = [g['group_id'] for g in matched_grouped]

        # Convert to polygons/multipolygons
        try:
            polygons = [shapes_to_polygon(g) for g in matched_grouped]
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
        # Note: Output is always a single polygon even if inputs had multiple parts (group_id)
        # because merge functions combine all parts into one
        group_id_info = ""
        if any(gid is not None for gid in group_ids):
            group_id_info = f" (from grouped shapes: {group_ids})"

        merged_shapes.append({
            'label': final_label,
            'points': merged_points,
            'group_id': None,
            'shape_type': 'polygon',
            'flags': {},
            'description': f'Merged obj {match_idx+1} from 3 annotators using {merge_method}{group_id_info}',
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
