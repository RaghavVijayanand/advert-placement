"""
Train LightGBM ranking model for ad placement candidate scoring.

This module extracts features from synthetic dataset and trains a pairwise
ranking model to predict the best ad placement candidates.
"""

import json
import os
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import cv2
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

from src.config import MIN_CONTRAST_RATIO, SALIENCY_THRESHOLD
from src.saliency import SaliencyEstimator
from src.detectors import ProtectedContentDetector


class RankerTrainer:
    """Train and evaluate LightGBM ranking model."""
    
    def __init__(self, data_dir: str = "data/synth", verbose: bool = True):
        """
        Initialize trainer.
        
        Args:
            data_dir: Directory containing synthetic posters + metadata.json
            verbose: Whether to print progress information
        """
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        
        # Initialize feature extractors
        self.saliency_estimator = SaliencyEstimator()
        self.detector = ProtectedContentDetector()
        
        # Storage
        self.features = []
        self.labels = []
        self.groups = []
        self.scaler = StandardScaler()
        
        if self.verbose:
            print(f"✓ Ranker trainer initialized")
            print(f"  Data directory: {self.data_dir}")
    
    def extract_features(self, image_path: str, candidate: Dict) -> np.ndarray:
        """
        Extract 15 features for a single candidate region.
        
        Features:
        1-4: Saliency stats (mean, max, min, std)
        5-6: Distance from protected content (mean, min)
        7-8: Edge proximity (mean, min distance to edges)
        9-10: Composition scores (rule of thirds, center bias)
        11: Aspect ratio deviation from standard ratios
        12: Area ratio (candidate area / image area)
        13-14: Texture complexity (entropy, variance)
        15: Plane confidence (if available, else 0)
        
        Args:
            image_path: Path to the poster image
            candidate: Dict with x, y, w, h keys
            
        Returns:
            Feature vector of shape (15,)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        H, W = image.shape[:2]
        x, y, w, h = candidate['x'], candidate['y'], candidate['w'], candidate['h']
        
        # Ensure valid bounds
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))
        
        features = []
        
        # 1-4: Saliency features
        saliency_map = self.saliency_estimator.estimate(image)
        roi_saliency = saliency_map[y:y+h, x:x+w]
        features.extend([
            float(np.mean(roi_saliency)),
            float(np.max(roi_saliency)),
            float(np.min(roi_saliency)),
            float(np.std(roi_saliency))
        ])
        
        # 5-6: Distance from protected content
        detection_result = self.detector.detect(image)
        protected_mask = detection_result['combined_mask']
        if np.any(protected_mask > 0):
            dist_transform = cv2.distanceTransform(
                (255 - protected_mask).astype(np.uint8), 
                cv2.DIST_L2, 5
            )
            roi_dist = dist_transform[y:y+h, x:x+w]
            features.extend([
                float(np.mean(roi_dist)),
                float(np.min(roi_dist))
            ])
        else:
            features.extend([float(max(W, H)), float(max(W, H))])
        
        # 7-8: Edge proximity
        edge_distances = np.array([
            x,              # left edge
            y,              # top edge
            W - (x + w),    # right edge
            H - (y + h)     # bottom edge
        ])
        features.extend([
            float(np.mean(edge_distances)),
            float(np.min(edge_distances))
        ])
        
        # 9-10: Composition scores
        cx, cy = x + w / 2, y + h / 2
        third_w, third_h = W / 3, H / 3
        
        # Rule of thirds score (distance to nearest intersection)
        intersections = [
            (third_w, third_h), (2*third_w, third_h),
            (third_w, 2*third_h), (2*third_w, 2*third_h)
        ]
        min_dist = min(
            np.sqrt((cx - ix)**2 + (cy - iy)**2) 
            for ix, iy in intersections
        )
        rule_of_thirds_score = 1.0 - min(min_dist / (W * 0.5), 1.0)
        
        # Center bias (inverse distance from center)
        center_dist = np.sqrt((cx - W/2)**2 + (cy - H/2)**2)
        max_dist = np.sqrt((W/2)**2 + (H/2)**2)
        center_bias_score = 1.0 - (center_dist / max_dist)
        
        features.extend([
            float(rule_of_thirds_score),
            float(center_bias_score)
        ])
        
        # 11: Aspect ratio deviation
        aspect = w / h
        standard_aspects = [1.0, 4/3, 3/2, 16/9, 9/16]
        aspect_dev = min(abs(aspect - std_asp) for std_asp in standard_aspects)
        features.append(float(aspect_dev))
        
        # 12: Area ratio
        area_ratio = (w * h) / (W * H)
        features.append(float(area_ratio))
        
        # 13-14: Texture complexity
        roi = image[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Entropy
        hist = cv2.calcHist([gray_roi], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        features.append(float(entropy))
        
        # Variance
        variance = float(np.var(gray_roi))
        features.append(variance)
        
        # 15: Plane confidence (placeholder - would need depth module)
        features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def load_dataset(self):
        """
        Load synthetic dataset and extract features.
        
        For each poster:
        - Safe regions get label = 1 (positive)
        - Random non-overlapping regions get label = 0 (negative)
        - Each poster forms a "query group" for ranking
        """
        metadata_path = self.data_dir / "metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if self.verbose:
            print(f"Loading {len(metadata)} synthetic posters...")
        
        for i, item in enumerate(metadata):
            image_path = str(self.data_dir / item['image'])
            
            if not os.path.exists(image_path):
                if self.verbose:
                    print(f"  Warning: Image not found: {image_path}")
                continue
            
            W, H = item['size']['width'], item['size']['height']
            
            # Positive samples (safe regions)
            safe_regions = item.get('safe_regions', [])
            for region in safe_regions[:10]:  # Limit to 10 per image
                try:
                    feat = self.extract_features(image_path, region)
                    self.features.append(feat)
                    self.labels.append(1)  # Safe region
                    self.groups.append(i)
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: Feature extraction failed: {e}")
            
            # Negative samples (random regions avoiding protected areas)
            # Generate 2x as many negatives as positives
            num_negatives = min(len(safe_regions) * 2, 20)
            
            for _ in range(num_negatives):
                # Random candidate
                aspect = np.random.choice([1.0, 4/3, 3/2, 16/9])
                area = np.random.uniform(0.02, 0.15) * W * H
                w = int(np.sqrt(area * aspect))
                h = int(w / aspect)
                x = np.random.randint(0, max(1, W - w))
                y = np.random.randint(0, max(1, H - h))
                
                candidate = {'x': x, 'y': y, 'w': w, 'h': h}
                
                try:
                    feat = self.extract_features(image_path, candidate)
                    self.features.append(feat)
                    self.labels.append(0)  # Random region
                    self.groups.append(i)
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: Feature extraction failed: {e}")
            
            if self.verbose and (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(metadata)} images")
        
        # Convert to numpy arrays
        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int32)
        self.groups = np.array(self.groups, dtype=np.int32)
        
        if self.verbose:
            print(f"✓ Dataset loaded")
            print(f"  Total samples: {len(self.features)}")
            print(f"  Positive samples: {np.sum(self.labels == 1)}")
            print(f"  Negative samples: {np.sum(self.labels == 0)}")
            print(f"  Feature dimension: {self.features.shape[1]}")
    
    def train(self, output_path: str = "models/ranker.pkl"):
        """
        Train LightGBM ranker with pairwise ranking loss.
        
        Args:
            output_path: Path to save trained model and scaler
        """
        if len(self.features) == 0:
            raise ValueError("No training data. Call load_dataset() first.")
        
        # Normalize features
        self.features = self.scaler.fit_transform(self.features)
        
        # Split into train/val, preserving groups
        unique_groups = np.unique(self.groups)
        train_groups, val_groups = train_test_split(
            unique_groups, test_size=0.2, random_state=42
        )
        
        train_mask = np.isin(self.groups, train_groups)
        val_mask = np.isin(self.groups, val_groups)
        
        X_train = self.features[train_mask]
        y_train = self.labels[train_mask]
        groups_train = self.groups[train_mask]
        
        X_val = self.features[val_mask]
        y_val = self.labels[val_mask]
        groups_val = self.groups[val_mask]
        
        # Compute group sizes for LightGBM
        _, train_group_sizes = np.unique(groups_train, return_counts=True)
        _, val_group_sizes = np.unique(groups_val, return_counts=True)
        
        if self.verbose:
            print(f"Training LightGBM ranker...")
            print(f"  Train samples: {len(X_train)} ({len(train_groups)} groups)")
            print(f"  Val samples: {len(X_val)} ({len(val_groups)} groups)")
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(
            X_train, label=y_train, group=train_group_sizes
        )
        val_data = lgb.Dataset(
            X_val, label=y_val, group=val_group_sizes, reference=train_data
        )
        
        # LightGBM parameters for ranking
        params = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [3, 5],
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # Train model
        callbacks = [lgb.log_evaluation(period=10)] if self.verbose else []
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks
        )
        
        # Save model and scaler
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': self.scaler,
                'feature_names': [
                    'saliency_mean', 'saliency_max', 'saliency_min', 'saliency_std',
                    'protected_dist_mean', 'protected_dist_min',
                    'edge_dist_mean', 'edge_dist_min',
                    'rule_of_thirds', 'center_bias',
                    'aspect_dev', 'area_ratio',
                    'entropy', 'variance',
                    'plane_confidence'
                ]
            }, f)
        
        if self.verbose:
            print(f"✓ Model saved to {output_path}")
            
            # Print feature importance
            importance = model.feature_importance(importance_type='gain')
            feature_names = [
                'saliency_mean', 'saliency_max', 'saliency_min', 'saliency_std',
                'protected_dist_mean', 'protected_dist_min',
                'edge_dist_mean', 'edge_dist_min',
                'rule_of_thirds', 'center_bias',
                'aspect_dev', 'area_ratio',
                'entropy', 'variance',
                'plane_confidence'
            ]
            
            print("\nTop 5 features by importance:")
            sorted_idx = np.argsort(importance)[::-1]
            for i in sorted_idx[:5]:
                print(f"  {feature_names[i]}: {importance[i]:.1f}")
        
        return model


def demo_training():
    """Demo: Train ranker on synthetic dataset."""
    # Generate synthetic data first if needed
    synth_dir = Path("data/synth_demo")
    if not synth_dir.exists() or not (synth_dir / "metadata.json").exists():
        print("Generating synthetic dataset first...")
        from src.synth_data import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator(output_dir=str(synth_dir))
        generator.generate_dataset(num_images=20, verbose=True)
    
    # Train ranker
    trainer = RankerTrainer(data_dir=str(synth_dir), verbose=True)
    trainer.load_dataset()
    trainer.train(output_path="models/ranker.pkl")
    
    print("\n✓ Training demo complete")


if __name__ == "__main__":
    demo_training()
