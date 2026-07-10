"""
Band Line Detector for New-Pattern Island Images

Thin subclass of ``LineDetector`` used to detect horizontal slanted print
lines inside a single vertical band that has been cropped out of a larger
island image.

The only behavioral change is kernel scaling: ``LineDetector`` scales kernel
size and the number of vertical scan columns by the input width relative to an
ideal reference. A band crop is much narrower than the full image, which would
shrink the kernels and scan count and change detection sensitivity. This
subclass scales width-dependent parameters using the FULL image width
(``reference_width``) instead of the crop width, so per-band detection behaves
identically to full-image detection. Height-dependent behavior (``Y_DELTA``)
and slope validation are inherited unchanged because a band keeps full height.
"""

from utils.line_detector import LineDetector


class BandLineDetector(LineDetector):
    """LineDetector variant that scales kernels to the full image width."""

    def __init__(self, sensitivity='medium', reference_width=None):
        """Initialize the band line detector.

        Args:
            sensitivity: Detection sensitivity level ('low', 'medium', 'high').
            reference_width: Full (uncropped) image width used for width-based
                kernel scaling. When None, falls back to the crop width.
        """
        super().__init__(sensitivity)
        self.reference_width = reference_width

    def update_kernel_dimensions_for_image(self, image_width, image_height, debug=False):
        """Scale kernels using the full image width instead of the crop width.

        Args:
            image_width: Width of the current (cropped) band image.
            image_height: Height of the current band image (full height).
            debug: Whether to print scaling information.
        """
        width_for_scaling = self.reference_width or image_width
        super().update_kernel_dimensions_for_image(width_for_scaling, image_height, debug)
