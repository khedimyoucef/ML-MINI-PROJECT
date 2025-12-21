"""
Dataset Cleaning Tool

This module provides automated tools to clean and validate the grocery dataset:
1. Corrupted image detection
2. Duplicate image detection using perceptual hashing
3. Outlier/mislabeled image detection using feature-based analysis
4. HTML report generation for manual review
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import json
from datetime import datetime

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils import CLASS_NAMES, get_image_transform


def find_corrupted_images(data_dir: str, splits: List[str] = ['train', 'test', 'val']) -> List[str]:
    """
    Find images that cannot be properly loaded or are corrupted.
    
    Args:
        data_dir: Path to dataset root (e.g., DS2GROCERIES)
        splits: List of splits to check
        
    Returns:
        List of paths to corrupted images
    """
    print("\n" + "=" * 60)
    print("STEP 1: Finding Corrupted Images")
    print("=" * 60)
    
    corrupted = []
    data_path = Path(data_dir)
    
    # Count total images first
    total_images = 0
    for split in splits:
        split_path = data_path / split
        if split_path.exists():
            for class_name in CLASS_NAMES:
                class_dir = split_path / class_name
                if class_dir.exists():
                    total_images += len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
    
    print(f"Scanning {total_images} images for corruption...")
    
    pbar = tqdm(total=total_images, desc="Checking images")
    
    for split in splits:
        split_path = data_path / split
        if not split_path.exists():
            continue
            
        for class_name in CLASS_NAMES:
            class_dir = split_path / class_name
            if not class_dir.exists():
                continue
                
            for img_path in list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")):
                try:
                    # Try to open and verify the image
                    with Image.open(img_path) as img:
                        img.verify()
                    
                    # Also try to actually load it (verify doesn't always catch issues)
                    with Image.open(img_path) as img:
                        img.load()
                        # Check if image has valid dimensions
                        if img.size[0] < 10 or img.size[1] < 10:
                            corrupted.append(str(img_path))
                            
                except Exception as e:
                    corrupted.append(str(img_path))
                
                pbar.update(1)
    
    pbar.close()
    print(f"\nFound {len(corrupted)} corrupted images")
    
    return corrupted


def find_duplicate_images(
    data_dir: str, 
    splits: List[str] = ['train', 'test', 'val'],
    hash_size: int = 8,
    threshold: int = 5
) -> Dict[str, List[str]]:
    """
    Find duplicate or near-duplicate images using perceptual hashing.
    
    Args:
        data_dir: Path to dataset root
        splits: List of splits to check
        hash_size: Size of the perceptual hash (larger = more precise)
        threshold: Hamming distance threshold for considering duplicates
        
    Returns:
        Dictionary mapping hash strings to lists of image paths
    """
    print("\n" + "=" * 60)
    print("STEP 2: Finding Duplicate Images")
    print("=" * 60)
    
    try:
        import imagehash
    except ImportError:
        print("WARNING: imagehash library not installed.")
        print("Install it with: pip install imagehash")
        print("Skipping duplicate detection...")
        return {}
    
    hashes = defaultdict(list)
    data_path = Path(data_dir)
    
    # Count total images first
    total_images = 0
    for split in splits:
        split_path = data_path / split
        if split_path.exists():
            for class_name in CLASS_NAMES:
                class_dir = split_path / class_name
                if class_dir.exists():
                    total_images += len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.png")))
    
    print(f"Computing perceptual hashes for {total_images} images...")
    
    pbar = tqdm(total=total_images, desc="Hashing images")
    
    for split in splits:
        split_path = data_path / split
        if not split_path.exists():
            continue
            
        for class_name in CLASS_NAMES:
            class_dir = split_path / class_name
            if not class_dir.exists():
                continue
                
            for img_path in list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png")):
                try:
                    with Image.open(img_path) as img:
                        # Compute perceptual hash
                        h = imagehash.phash(img.convert('RGB'), hash_size=hash_size)
                        hashes[str(h)].append(str(img_path))
                except Exception:
                    pass  # Skip corrupted images (already handled)
                
                pbar.update(1)
    
    pbar.close()
    
    # Filter to only keep groups with duplicates
    duplicates = {k: v for k, v in hashes.items() if len(v) > 1}
    
    total_duplicate_images = sum(len(v) - 1 for v in duplicates.values())
    print(f"\nFound {len(duplicates)} groups of duplicates ({total_duplicate_images} redundant images)")
    
    return duplicates


def find_outliers_per_class(
    data_dir: str,
    split: str = 'train',
    model_path: Optional[str] = None,
    contamination: float = 0.05,
    n_neighbors: int = 20,
    device: Optional[str] = None
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Find outlier/mislabeled images using feature-based analysis.
    
    Uses Local Outlier Factor on ResNet50 features to identify
    images that don't fit well within their labeled class.
    
    Args:
        data_dir: Path to dataset root
        split: Which split to analyze
        model_path: Optional path to fine-tuned feature extractor
        contamination: Expected proportion of outliers (0.0 to 0.5)
        n_neighbors: Number of neighbors for LOF algorithm
        device: Device for feature extraction
        
    Returns:
        Dictionary mapping class names to lists of (path, outlier_score) tuples
    """
    print("\n" + "=" * 60)
    print("STEP 3: Finding Outliers/Mislabeled Images")
    print("=" * 60)
    
    from sklearn.neighbors import LocalOutlierFactor
    from src.feature_extraction import FeatureExtractor
    from src.data_utils import GroceryDataset
    
    # Initialize feature extractor
    print("\nLoading feature extractor...")
    extractor = FeatureExtractor(device=device, model_path=model_path)
    
    # Load dataset
    print(f"Loading {split} dataset...")
    dataset = GroceryDataset(data_dir, split=split, transform=get_image_transform())
    
    # Extract features
    print("Extracting features (this may take a while)...")
    features, labels = extractor.extract_batch(dataset, batch_size=16)
    paths = dataset.get_paths()
    
    # Find outliers per class
    print("\nRunning outlier detection per class...")
    outliers_by_class = {}
    
    for class_idx, class_name in enumerate(tqdm(CLASS_NAMES, desc="Analyzing classes")):
        # Get indices for this class
        class_mask = labels == class_idx
        n_class_samples = class_mask.sum()
        
        if n_class_samples < n_neighbors + 1:
            print(f"  Skipping {class_name}: not enough samples ({n_class_samples})")
            continue
        
        class_features = features[class_mask]
        class_indices = np.where(class_mask)[0]
        
        # Run Local Outlier Factor
        lof = LocalOutlierFactor(
            n_neighbors=min(n_neighbors, n_class_samples - 1),
            contamination=contamination,
            novelty=False
        )
        
        # Fit and predict
        predictions = lof.fit_predict(class_features)
        scores = -lof.negative_outlier_factor_  # Higher = more outlier-like
        
        # Collect outliers (prediction == -1)
        outlier_mask = predictions == -1
        outlier_indices = class_indices[outlier_mask]
        outlier_scores = scores[outlier_mask]
        
        # Sort by score (most anomalous first)
        sorted_indices = np.argsort(outlier_scores)[::-1]
        
        outliers = []
        for idx in sorted_indices:
            original_idx = outlier_indices[idx]
            outliers.append((paths[original_idx], float(outlier_scores[idx])))
        
        if outliers:
            outliers_by_class[class_name] = outliers
    
    total_outliers = sum(len(v) for v in outliers_by_class.values())
    print(f"\nFound {total_outliers} potential outliers across {len(outliers_by_class)} classes")
    
    return outliers_by_class


def generate_html_report(
    corrupted: List[str],
    duplicates: Dict[str, List[str]],
    outliers: Dict[str, List[Tuple[str, float]]],
    output_path: str = "dataset_cleaning_report.html",
    max_display_per_category: int = 50
) -> str:
    """
    Generate an HTML report for reviewing flagged images.
    
    Args:
        corrupted: List of corrupted image paths
        duplicates: Dictionary of duplicate groups
        outliers: Dictionary of outliers by class
        output_path: Path to save the HTML report
        max_display_per_category: Maximum images to display per category
        
    Returns:
        Path to the generated report
    """
    print("\n" + "=" * 60)
    print("STEP 4: Generating HTML Report")
    print("=" * 60)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Dataset Cleaning Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #1a1a2e;
            color: #eaeaea;
        }}
        h1 {{
            color: #00d9ff;
            border-bottom: 2px solid #00d9ff;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #ff6b6b;
            margin-top: 40px;
        }}
        h3 {{
            color: #ffd93d;
        }}
        .summary {{
            background: #16213e;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
        }}
        .stat-box {{
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-number {{
            font-size: 2rem;
            font-weight: bold;
            color: #00d9ff;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 10px;
            margin: 20px 0;
        }}
        .image-card {{
            background: #16213e;
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.2s;
        }}
        .image-card:hover {{
            transform: scale(1.05);
        }}
        .image-card img {{
            width: 100%;
            height: 120px;
            object-fit: cover;
        }}
        .image-card .path {{
            padding: 8px;
            font-size: 0.7rem;
            word-break: break-all;
            color: #888;
        }}
        .image-card .score {{
            padding: 5px 8px;
            background: #ff6b6b;
            color: white;
            font-size: 0.75rem;
        }}
        .duplicate-group {{
            background: #16213e;
            border-radius: 10px;
            padding: 15px;
            margin: 15px 0;
        }}
        .action-buttons {{
            margin: 20px 0;
        }}
        button {{
            background: #00d9ff;
            color: #1a1a2e;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin-right: 10px;
            font-weight: bold;
        }}
        button:hover {{
            background: #00b8d4;
        }}
        .file-list {{
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            max-height: 300px;
            overflow-y: auto;
            font-family: monospace;
            font-size: 0.8rem;
        }}
        .collapsible {{
            cursor: pointer;
            padding: 10px;
            background: #16213e;
            border-radius: 5px;
            margin: 5px 0;
        }}
        .collapsible:hover {{
            background: #1a2744;
        }}
        .content {{
            display: none;
            padding: 10px;
        }}
    </style>
</head>
<body>
    <h1>🧹 Dataset Cleaning Report</h1>
    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <h2 style="margin-top: 0;">📊 Summary</h2>
        <div class="summary-grid">
            <div class="stat-box">
                <div class="stat-number">{len(corrupted)}</div>
                <div>Corrupted Images</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{sum(len(v) - 1 for v in duplicates.values())}</div>
                <div>Duplicate Images</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{sum(len(v) for v in outliers.values())}</div>
                <div>Potential Outliers</div>
            </div>
        </div>
    </div>
"""
    
    # Corrupted Images Section
    html += """
    <h2>❌ Corrupted Images</h2>
    <p>These images failed to load or have invalid formats. They should be deleted.</p>
"""
    if corrupted:
        html += f"""
    <div class="action-buttons">
        <button onclick="copyToClipboard('corrupted-list')">📋 Copy File List</button>
    </div>
    <div class="file-list" id="corrupted-list">
"""
        for path in corrupted[:max_display_per_category]:
            html += f"{path}\n"
        if len(corrupted) > max_display_per_category:
            html += f"\n... and {len(corrupted) - max_display_per_category} more"
        html += "</div>"
    else:
        html += "<p>✅ No corrupted images found!</p>"
    
    # Duplicates Section
    html += """
    <h2>🔁 Duplicate Images</h2>
    <p>These images are duplicates or near-duplicates. Keep one from each group and delete the rest.</p>
"""
    if duplicates:
        displayed = 0
        for hash_val, paths in list(duplicates.items())[:20]:
            html += f"""
    <div class="duplicate-group">
        <strong>Group ({len(paths)} images)</strong>
        <div class="image-grid">
"""
            for path in paths[:5]:
                # Convert to file:// URL for local display
                file_url = Path(path).resolve().as_uri()
                html += f"""
            <div class="image-card">
                <img src="{file_url}" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22><text y=%2250%%22>❌</text></svg>'">
                <div class="path">{Path(path).name}</div>
            </div>
"""
            if len(paths) > 5:
                html += f"<div class='image-card'><div class='path'>+{len(paths)-5} more</div></div>"
            html += "</div></div>"
            displayed += 1
        
        if len(duplicates) > 20:
            html += f"<p>... and {len(duplicates) - 20} more duplicate groups</p>"
    else:
        html += "<p>✅ No duplicate images found!</p>"
    
    # Outliers Section
    html += """
    <h2>⚠️ Potential Outliers / Mislabeled Images</h2>
    <p>These images may be mislabeled or don't fit their class well. Review and relabel or delete them.</p>
"""
    if outliers:
        for class_name, class_outliers in outliers.items():
            html += f"""
    <div class="collapsible" onclick="toggleContent('{class_name}')">
        📁 {class_name} ({len(class_outliers)} outliers) ▼
    </div>
    <div class="content" id="{class_name}">
        <div class="image-grid">
"""
            for path, score in class_outliers[:max_display_per_category]:
                file_url = Path(path).resolve().as_uri()
                html += f"""
            <div class="image-card">
                <img src="{file_url}" onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22><text y=%2250%%22>❌</text></svg>'">
                <div class="score">Score: {score:.2f}</div>
                <div class="path">{Path(path).name}</div>
            </div>
"""
            html += "</div></div>"
    else:
        html += "<p>✅ No significant outliers detected!</p>"
    
    # JavaScript
    html += """
    <script>
        function toggleContent(id) {
            var content = document.getElementById(id);
            content.style.display = content.style.display === 'block' ? 'none' : 'block';
        }
        
        function copyToClipboard(id) {
            var text = document.getElementById(id).innerText;
            navigator.clipboard.writeText(text).then(function() {
                alert('Copied to clipboard!');
            });
        }
    </script>
</body>
</html>
"""
    
    # Write the report
    # Make sure the output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Report saved to: {output_path}")
    return output_path


def save_file_lists(
    corrupted: List[str],
    duplicates: Dict[str, List[str]],
    outliers: Dict[str, List[Tuple[str, float]]],
    output_dir: str = "cleaning_results"
) -> Dict[str, str]:
    """
    Save lists of flagged files for easy batch operations.
    
    Args:
        corrupted: List of corrupted image paths
        duplicates: Dictionary of duplicate groups
        outliers: Dictionary of outliers by class
        output_dir: Directory to save the file lists
        
    Returns:
        Dictionary mapping list names to file paths
    """
    print("\n" + "=" * 60)
    print("STEP 5: Saving File Lists")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    
    # Save corrupted files list
    corrupted_path = output_path / "corrupted_images.txt"
    with open(corrupted_path, 'w') as f:
        for path in corrupted:
            f.write(f"{path}\n")
    saved_files['corrupted'] = str(corrupted_path)
    print(f"  Corrupted images list: {corrupted_path}")
    
    # Save duplicate files list (all except first in each group)
    duplicates_to_delete = []
    for paths in duplicates.values():
        # Keep first, mark rest for deletion
        duplicates_to_delete.extend(paths[1:])
    
    duplicates_path = output_path / "duplicate_images.txt"
    with open(duplicates_path, 'w') as f:
        for path in duplicates_to_delete:
            f.write(f"{path}\n")
    saved_files['duplicates'] = str(duplicates_path)
    print(f"  Duplicate images list: {duplicates_path}")
    
    # Save outliers list
    all_outliers = []
    for class_name, class_outliers in outliers.items():
        for path, score in class_outliers:
            all_outliers.append(f"{path}\t{class_name}\t{score:.3f}")
    
    outliers_path = output_path / "outlier_images.txt"
    with open(outliers_path, 'w') as f:
        f.write("# path\tclass\toutlier_score\n")
        for line in all_outliers:
            f.write(f"{line}\n")
    saved_files['outliers'] = str(outliers_path)
    print(f"  Outlier images list: {outliers_path}")
    
    # Save combined "safe to delete" list (corrupted + duplicates)
    safe_to_delete = corrupted + duplicates_to_delete
    delete_path = output_path / "safe_to_delete.txt"
    with open(delete_path, 'w') as f:
        for path in safe_to_delete:
            f.write(f"{path}\n")
    saved_files['safe_to_delete'] = str(delete_path)
    print(f"  Safe to delete list: {delete_path}")
    
    # Save summary JSON
    summary = {
        'timestamp': datetime.now().isoformat(),
        'counts': {
            'corrupted': len(corrupted),
            'duplicates': len(duplicates_to_delete),
            'outliers': len(all_outliers),
            'safe_to_delete': len(safe_to_delete)
        },
        'files': saved_files
    }
    
    summary_path = output_path / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary: {summary_path}")
    
    return saved_files


def clean_dataset(
    data_dir: str = "DS2GROCERIES",
    output_dir: str = "cleaning_results",
    model_path: Optional[str] = None,
    contamination: float = 0.05,
    splits: List[str] = ['train'],
    device: Optional[str] = None,
    skip_duplicates: bool = False,
    skip_outliers: bool = False
) -> Dict:
    """
    Run the complete dataset cleaning pipeline.
    
    Args:
        data_dir: Path to dataset root
        output_dir: Directory to save results
        model_path: Path to fine-tuned feature extractor (optional)
        contamination: Expected proportion of outliers
        splits: Splits to analyze
        device: Device for feature extraction
        skip_duplicates: Skip duplicate detection
        skip_outliers: Skip outlier detection
        
    Returns:
        Dictionary with all results
    """
    print("\n" + "=" * 70)
    print("DATASET CLEANING PIPELINE")
    print("=" * 70)
    print(f"Dataset: {data_dir}")
    print(f"Splits: {splits}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    results = {}
    
    # Step 1: Find corrupted images
    corrupted = find_corrupted_images(data_dir, splits)
    results['corrupted'] = corrupted
    
    # Step 2: Find duplicates
    if skip_duplicates:
        print("\nSkipping duplicate detection...")
        duplicates = {}
    else:
        duplicates = find_duplicate_images(data_dir, splits)
    results['duplicates'] = duplicates
    
    # Step 3: Find outliers
    if skip_outliers:
        print("\nSkipping outlier detection...")
        outliers = {}
    else:
        outliers = find_outliers_per_class(
            data_dir,
            split=splits[0],  # Only analyze first split for outliers
            model_path=model_path,
            contamination=contamination,
            device=device
        )
    results['outliers'] = outliers
    
    # Step 4: Generate HTML report
    report_path = os.path.join(output_dir, "dataset_cleaning_report.html")
    generate_html_report(corrupted, duplicates, outliers, report_path)
    results['report'] = report_path
    
    # Step 5: Save file lists
    file_lists = save_file_lists(corrupted, duplicates, outliers, output_dir)
    results['files'] = file_lists
    
    # Print summary
    print("\n" + "=" * 70)
    print("CLEANING COMPLETE!")
    print("=" * 70)
    print(f"\nResults Summary:")
    print(f"   - Corrupted images: {len(corrupted)}")
    print(f"   - Duplicate groups: {len(duplicates)}")
    print(f"   - Classes with outliers: {len(outliers)}")
    print(f"   - Total outliers: {sum(len(v) for v in outliers.values())}")
    print(f"\nOutput files saved to: {output_dir}/")
    print(f"   - HTML Report: {report_path}")
    print(f"\nNext Steps:")
    print(f"   1. Open the HTML report in your browser to review flagged images")
    print(f"   2. Delete files listed in 'safe_to_delete.txt' (corrupted + duplicates)")
    print(f"   3. Manually review outliers and relabel or delete as needed")
    print(f"   4. Re-train your model on the cleaned dataset")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Clean and validate the grocery dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full cleaning pipeline
  python dataset_cleaner.py --data-dir DS2GROCERIES
  
  # Quick scan (skip slow outlier detection)
  python dataset_cleaner.py --data-dir DS2GROCERIES --skip-outliers
  
  # Use fine-tuned model for better outlier detection
  python dataset_cleaner.py --data-dir DS2GROCERIES --model-path models/feature_extractor.pth
  
  # Higher contamination rate (find more outliers)
  python dataset_cleaner.py --data-dir DS2GROCERIES --contamination 0.1
        """
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='DS2GROCERIES',
        help='Path to dataset directory (default: DS2GROCERIES)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='cleaning_results',
        help='Directory to save cleaning results (default: cleaning_results)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to fine-tuned feature extractor (optional, uses pretrained if not provided)'
    )
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.05,
        help='Expected proportion of outliers per class (default: 0.05 = 5%%)'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train'],
        help='Splits to analyze (default: train)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device for feature extraction (cuda/cpu/auto)'
    )
    parser.add_argument(
        '--skip-duplicates',
        action='store_true',
        help='Skip duplicate detection (faster)'
    )
    parser.add_argument(
        '--skip-outliers',
        action='store_true',
        help='Skip outlier detection (much faster)'
    )
    
    args = parser.parse_args()
    
    clean_dataset(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        contamination=args.contamination,
        splits=args.splits,
        device=args.device,
        skip_duplicates=args.skip_duplicates,
        skip_outliers=args.skip_outliers
    )


if __name__ == "__main__":
    main()
