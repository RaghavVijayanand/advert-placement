"""Configuration and constants for the ad placement system."""

# Image processing
MIN_IMAGE_SIZE = 512  # minimum edge length in pixels
MAX_IMAGE_SIZE = 8192  # maximum edge length
PROCESS_SIZE = 1024  # processing resolution (short side)

# Candidate constraints
MIN_AD_AREA_PCT = 0.01  # 1% of image area
MAX_AD_AREA_PCT = 0.12  # 12% of image area
ASPECT_RATIOS = [(1, 1), (4, 3), (3, 2), (16, 9), (9, 16)]  # allowed aspect ratios
PADDING_PX = 24  # minimum padding from protected content
PADDING_PCT = 0.01  # padding as % of min edge

# Saliency
SALIENCY_THRESHOLD = 0.8  # top 20% saliency considered "high attention"
MAX_SALIENCY_OVERLAP = 0.05  # max 5% overlap with high-saliency region

# Candidate generation
MAX_RAW_CANDIDATES = 300
MAX_FILTERED_CANDIDATES = 30
TOP_K_CANDIDATES = 5

# Detection thresholds
FACE_CONF_THRESHOLD = 0.5
TEXT_CONF_THRESHOLD = 0.5
OBJECT_CONF_THRESHOLD = 0.3

# Compositing
MIN_CONTRAST_RATIO = 4.5  # WCAG AA standard
DEFAULT_SHADOW_OPACITY = 0.3
DEFAULT_BORDER_WIDTH = 2

# Model paths (relative to project root)
MODELS_DIR = "models"
CACHE_DIR = "models/cache"
