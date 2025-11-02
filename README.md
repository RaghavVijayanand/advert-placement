# Discreet Ad Placement & Auto-Compositing (DAPAC)

Automatically find and place advertisements discreetly on poster images without affecting core content.

## Features
- ✅ **Context-aware placement** - Automatically detects natural ad surfaces (TV screens, wall posters, picture frames)
- ✅ Protected content detection (faces, text regions)
- ✅ Saliency-aware placement suggestions
- ✅ Depth estimation and planar surface detection
- ✅ Semantic segmentation for uniform backgrounds
- ✅ LightGBM-based learned ranker (optional, trained on synthetic data)
- ✅ Automatic ad asset compositing with validation
- ✅ Synthetic poster dataset generator with ground truth
- ✅ Zero training data required (heuristic-based v0)
- ✅ Fast processing (~0.6-0.7s for 800x600 on CPU)
- ✅ CLI interface for batch processing

## Installation

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Analyze poster and find candidate placements

```powershell
python main.py analyze --input poster.jpg --save-overlays --verbose
```

This will:
- Detect protected content (faces, text)
- Generate candidate ad placement regions
- Rank them by discreetness score
- Save results JSON and visualization overlays to `outputs/`

**Output:**
- `poster_analysis.json` - Candidate placements with scores
- `poster_candidates.jpg` - Visual overlay showing top 5 candidates
- `poster_protected_mask.png` - Protected regions mask
- `poster_saliency.png` - Saliency heatmap

### 2. Composite ad onto selected candidate

```powershell
python main.py compose --input poster.jpg --ad my_ad.png --candidate 0 --output final.jpg --verbose
```

This will:
- Run analysis to find candidates
- Select candidate #0 (best-ranked)
- Auto-composite ad with proper scaling, borders, and shadows
- Validate no overlap with protected content
- Save final composite

## Architecture

### Detection (`src/detectors.py`)
- **Faces**: OpenCV Haar Cascade (basic fallback)
- **Text**: Edge-based heuristic detector
- **Protected regions**: Dilated masks with configurable padding

### Saliency (`src/saliency.py`)
- **Method**: Spectral residual + edge density + color contrast + center bias
- **Output**: Normalized saliency map [0, 1]

### Depth & Planes (`src/depth_planes.py`)
- **Depth**: Heuristic estimation (vertical position 50% + sharpness 30% + saturation 20%)
- **Planes**: DBSCAN clustering on depth+position coordinates
- **Output**: Depth map, plane masks, normals, confidence scores

### Segmentation (`src/segmentation.py`)
- **Method**: K-means (5 clusters) on LAB color + spatial position
- **Classification**: Heuristic labeling (sky/wall/floor based on position + color)
- **Uniformity**: Local variance analysis (inverse std dev)
- **Output**: Segmentation map, class masks, uniform regions

### Context Detection (`src/context_detector.py`) **NEW!**
- **Purpose**: Find natural ad placement surfaces
- **Detects**:
  - **TV/Monitor screens** (blue glow, 16:9 aspect, 99% confidence on test)
  - **Wall poster areas** (uniform backgrounds, good lighting)
  - **Picture frames** (rectangular borders on walls)
  - **Billboards** (outdoor large surfaces)
  - **Vehicle sides** (buses, trucks)
- **Method**: Color analysis (HSV), brightness detection, contour rectangularity
- **Priority**: Context surfaces get +15-25% score bonus in ranking

### Candidate Generation (`src/candidate_generator.py`)
- **Strategy 1**: Context-aware surface detection (TVs, frames, walls)
- **Strategy 2**: Random sampling with aspect ratio constraints
- **Constraints**: 
  - Area: 1-12% of image
  - Aspect ratios: 1:1, 4:3, 3:2, 16:9, 9:16
  - Zero overlap with protected regions
- **NMS**: Reduces overlapping candidates (IoU threshold 0.3)

### Ranking (`src/ranker.py` + `src/train_ranker.py`)
- **Heuristic Mode** (default):
  - Saliency overlap (35% weight)
  - Distance from protected content (25%)
  - Edge distance (15%)
  - Composition score (rule of thirds, 15%)
  - Aspect ratio match (10%)
- **Learned Mode** (with trained model):
  - LightGBM pairwise ranker (LambdaRank)
  - 15 features: saliency stats, protected distance, edge proximity, composition, texture
  - Trained on synthetic dataset (NDCG@3=1.0, NDCG@5=0.79)
  - Uses trained model from `models/ranker.pkl`
- **Output**: Top 5 ranked candidates

### Compositing (`src/compositor.py`)
- **Scaling**: Fit-to-region with margins
- **Legibility**: Auto-add white border if low contrast (<4.5:1)
- **Shadow**: Optional drop shadow for depth
- **Validation**: Zero overlap post-placement check

### Synthetic Data (`src/synth_data.py`)
- **Purpose**: Generate training data for LightGBM ranker
- **Content**: Random backgrounds with gradients + text/faces/logos
- **Ground truth**: Safe regions (non-overlapping with protected content)
- **Output**: Images + JSON metadata with bboxes and safe regions

## Performance

Tested on 800x600 test image (CPU):
- **Total time**: ~0.6-0.7s
- **Detection**: ~0.15s
- **Saliency**: ~0.25s
- **Generation**: ~0.05s
- **Ranking**: ~0.01s
- **Compositing**: ~0.15s

## Configuration

Edit `src/config.py` to customize:
- Min/max ad area percentage
- Allowed aspect ratios
- Padding around protected regions
- Saliency threshold
- Contrast ratio requirements

## Training the Learned Ranker

The system includes an optional LightGBM-based ranker that can be trained on synthetic data:

```powershell
# Generate synthetic dataset (20 images with ground truth)
python -c "from src.synth_data import SyntheticDataGenerator; gen = SyntheticDataGenerator('data/synth'); gen.generate_dataset(20)"

# Train LightGBM ranker
python -c "from src.train_ranker import demo_training; demo_training()"

# The trained model is saved to models/ranker.pkl
# The pipeline will automatically use it if available
```

The learned ranker achieves **NDCG@3 = 1.0** and **NDCG@5 = 0.79** on validation data.

## Future Improvements (v2)

- Replace heuristic detectors with deep models:
  - Text: PaddleOCR with PaddlePaddle backend
  - Faces: RetinaFace or MTCNN
  - People: DeepLabv3 or SegFormer
  - Saliency: U^2-Net or BASNet
  - Depth: MiDaS or DPT for better depth estimation
- Perspective-aware compositing using homography transforms
- Real labeled dataset collection for ranking
- GPU acceleration (DirectML provider available)

## Known Limitations

- Text detection is basic (edge-based heuristic)
- Face detection uses simple Haar cascade
- Depth estimation is heuristic-based (not deep learning)
- Segmentation is K-means clustering (not semantic understanding)
- Perspective correction is basic (homography only)
- No object detection beyond faces/text

## License

MIT License. Dependency licenses:
- OpenCV: Apache 2.0
- PyTorch: BSD-3
- NumPy: BSD
- timm: Apache 2.0
- LightGBM: MIT

## Examples

See `outputs/` folder after running demo for example inputs and results.

## Development

Run module tests:
```powershell
python src\test_models.py
python -c "from src.detectors import test_detector; test_detector()"
python -c "from src.saliency import test_saliency; test_saliency()"
python -c "from src.depth_planes import test_depth_planes; test_depth_planes()"
python -c "from src.segmentation import test_segmentation; test_segmentation()"
python -c "from src.candidate_generator import test_generator; test_generator()"
python -c "from src.ranker import test_ranker; test_ranker()"
python -c "from src.compositor import test_compositor; test_compositor()"
python -c "from src.pipeline import demo_pipeline; demo_pipeline()"
```

Generate synthetic data:
```powershell
python -c "from src.synth_data import demo_synthetic_generator; demo_synthetic_generator()"
```

Train LightGBM ranker:
```powershell
python -c "from src.train_ranker import demo_training; demo_training()"
```

## Contact

For issues or feature requests, please open an issue on GitHub.
