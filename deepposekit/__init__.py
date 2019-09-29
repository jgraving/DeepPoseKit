from __future__ import absolute_import
import sys
import warnings

from deepposekit.io import TrainingGenerator, DataGenerator
from deepposekit.augment.FlipAxis import FlipAxis

major = sys.version_info.major
minor = sys.version_info.minor

if major >= 3 and minor >= 6:
    ImportError = ModuleNotFoundError

try:
    import dpk_annotator as annotation

    Annotator = annotation.Annotator
    Skeleton = annotation.Skeleton
    KMeansSampler = annotation.KMeansSampler
    VideoReader = annotation.VideoReader
    VideoWriter = annotation.VideoWriter

except ImportError:
    warnings.warn(
        "\n"
        "\nDeepPoseKit Annotator is not found. "
        "\nAnnotation functions are not available. "
        "\nSee https://github.com/jgraving/deepposekit-annotator for installation instructions. "
        "\n"
    )


__version__ = "0.2.1"
