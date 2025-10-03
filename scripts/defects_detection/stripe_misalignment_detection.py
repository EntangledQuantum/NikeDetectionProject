"""
Stripe Misalignment Detection Algorithm
Detects misalignment of vertical stripes by analyzing X-coordinate trends

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from utils.image_saver import save_image
from utils.vertical_line_detector import VerticalLineDetector


class StripeMisalignmentDetector:
    """Detect vertical stripe misalignment via kernel scanning and edge preprocessing.

    The algorithm enhances vertical edges, then scans rows with a rectangular
    kernel to find the first strong vertical line per row. It compares the x
    positions across rows to flag significant lateral shifts as defects.
    Sensitivity presets adjust kernel shape, step size, and thresholds.
    """
    
    def __init__(self, sensitivity='medium', debug=False):
        """Configure misalignment detection using vertical line detector.

        Args:
            sensitivity: One of {'low', 'medium', 'high'}.
            debug: If True, draw visualization and store debug images.
        """
        self.debug = debug
        self.sensitivity = sensitivity
        
        print(f"Stripe Misalignment Sensitivity: {sensitivity}")
        
        # Initialize vertical line detector
        self.vertical_line_detector = VerticalLineDetector(sensitivity)
        
        # Thresholds for detecting misalignment trends
        self.min_x_deviation = 10  # Minimum X deviation to consider (same as min_x_shift_threshold)
        self.max_x_deviation = 500  # Maximum X deviation before ignoring as noise
        self.min_trend_rows = 3  # Minimum consecutive rows showing same trend to mark as defect
        
        print(f"Misalignment Detection Parameters:")
        print(f"  min_x_deviation: {self.min_x_deviation}px (minimum shift to consider)")
        print(f"  max_x_deviation: {self.max_x_deviation}px (maximum shift before ignoring)")
        print(f"  min_trend_rows: {self.min_trend_rows} (consecutive rows with trend to mark defect)")
    
    def analyze_x_trends(self, left_points, right_points, debug=False):
        """Analyze X-coordinate trends to detect misalignment
        
        Looks for continuous increase or decrease in X positions across rows
        
        Args:
            left_points: Left edge points sorted by Y
            right_points: Right edge points sorted by Y
            debug: Print debug info
            
        Returns:
            tuple: (defects, trend_data) - defects list and data for graphing
        """
        defects = []
        
        # Sort by Y
        left_sorted = sorted(left_points, key=lambda p: p['y'])
        right_sorted = sorted(right_points, key=lambda p: p['y'])
        
        min_len = min(len(left_sorted), len(right_sorted))
        if min_len < 2:
            return defects, None
        
        # Store data for graphing
        y_positions = []
        left_x_positions = []
        right_x_positions = []
        avg_x_positions = []
        trend_markers = []  # Store which rows are part of defect trends
        
        # Track trends
        trend_start_idx = 0
        trend_direction = None  # 'increase' or 'decrease'
        consecutive_trend_rows = 0
        current_trend_rows = []  # Track row indices in current trend
        
        for i in range(min_len):
            left_x = left_sorted[i]['x']
            right_x = right_sorted[i]['x']
            y = left_sorted[i]['y']
            avg_x = (left_x + right_x) / 2
            
            y_positions.append(y)
            left_x_positions.append(left_x)
            right_x_positions.append(right_x)
            avg_x_positions.append(avg_x)
            trend_markers.append(False)  # Default: not part of defect
            
            if i == 0:
                continue
            
            prev_left_x = left_sorted[i-1]['x']
            prev_right_x = right_sorted[i-1]['x']
            
            # Calculate X shift
            left_shift = left_x - prev_left_x
            right_shift = right_x - prev_right_x
            avg_shift = (left_shift + right_shift) / 2
            abs_shift = abs(avg_shift)
            
            if debug and i < 20:  # Print first 20 for debugging
                print(f"  Row {i}: Y={y}, AvgX={avg_x:.1f}, Shift={avg_shift:.1f}px")
            
            # Ignore if shift is too large (noise)
            if abs_shift > self.max_x_deviation:
                # Reset trend
                if debug:
                    print(f"    RESET: shift {abs_shift:.1f} > max {self.max_x_deviation}")
                trend_direction = None
                consecutive_trend_rows = 0
                current_trend_rows = []
                trend_start_idx = i
                continue
            
            # Check if shift is significant enough
            if abs_shift > self.min_x_deviation:
                current_direction = 'increase' if avg_shift > 0 else 'decrease'
                
                # Check if trend continues
                if trend_direction == current_direction:
                    consecutive_trend_rows += 1
                    current_trend_rows.append(i)
                else:
                    # New trend starting
                    trend_direction = current_direction
                    consecutive_trend_rows = 1
                    current_trend_rows = [i]
                    trend_start_idx = i
                
                if debug:
                    print(f"    Trend {trend_direction}: {consecutive_trend_rows} rows")
                
                # Mark as defect if trend persists for min_trend_rows
                if consecutive_trend_rows >= self.min_trend_rows:
                    # Mark all rows in this trend
                    for row_idx in current_trend_rows:
                        if row_idx < len(trend_markers):
                            trend_markers[row_idx] = True
                    
                    # Create defect for this misalignment region (only once)
                    if consecutive_trend_rows == self.min_trend_rows:
                        defect_y_start = left_sorted[trend_start_idx]['y']
                        defect_y_end = left_sorted[i]['y']
                        defect_x = avg_x
                        
                        defects.append({
                            'type': 'stripe_misalignment',
                            'y_start': defect_y_start,
                            'y_end': defect_y_end,
                            'y_center': (defect_y_start + defect_y_end) // 2,
                            'x': int(defect_x),
                            'trend': trend_direction,
                            'consecutive_rows': consecutive_trend_rows,
                            'avg_shift_per_row': abs_shift,
                            'location': (int(defect_x), (defect_y_start + defect_y_end) // 2),
                            'row_indices': list(current_trend_rows)
                        })
                        
                        if debug:
                            print(f"  ✓ DEFECT: {trend_direction} for {consecutive_trend_rows} rows, "
                                  f"Y=[{defect_y_start}-{defect_y_end}], shift={abs_shift:.1f}px/row")
            else:
                # Small shift - reset trend
                trend_direction = None
                consecutive_trend_rows = 0
                current_trend_rows = []
                trend_start_idx = i
        
        trend_data = {
            'y_positions': y_positions,
            'left_x': left_x_positions,
            'right_x': right_x_positions,
            'avg_x': avg_x_positions,
            'trend_markers': trend_markers
        }
        
        if debug:
            print(f"\nDetected {len(defects)} misalignment regions with sustained trends")
        
        return defects, trend_data
    
    def create_x_trend_graph(self, trend_data, output_dir, base_name):
        """Create graph showing X-coordinates over Y (rows)
        
        Args:
            trend_data: Dictionary with y_positions, left_x, right_x, avg_x, trend_markers
            output_dir: Directory to save graph
            base_name: Base filename
            
        Returns:
            Path to saved graph
        """
        if trend_data is None:
            return None
        
        y_pos = trend_data['y_positions']
        left_x = trend_data['left_x']
        right_x = trend_data['right_x']
        avg_x = trend_data['avg_x']
        markers = trend_data['trend_markers']
        
        # Create figure with larger size
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        
        # Plot 1: X coordinates over rows
        row_indices = list(range(len(y_pos)))
        
        # Normal points (not part of defect)
        normal_indices = [i for i, m in enumerate(markers) if not m]
        defect_indices = [i for i, m in enumerate(markers) if m]
        
        # Plot normal points
        if normal_indices:
            ax1.scatter([row_indices[i] for i in normal_indices],
                       [left_x[i] for i in normal_indices],
                       c='blue', s=10, alpha=0.5, label='Left edge (normal)')
            ax1.scatter([row_indices[i] for i in normal_indices],
                       [right_x[i] for i in normal_indices],
                       c='red', s=10, alpha=0.5, label='Right edge (normal)')
            ax1.plot([row_indices[i] for i in normal_indices],
                    [avg_x[i] for i in normal_indices],
                    'g-', linewidth=1, alpha=0.7, label='Average X (normal)')
        
        # Plot defect points in YELLOW
        if defect_indices:
            ax1.scatter([row_indices[i] for i in defect_indices],
                       [left_x[i] for i in defect_indices],
                       c='yellow', s=30, marker='o', edgecolors='black', linewidths=1,
                       label='Left edge (DEFECT)', zorder=5)
            ax1.scatter([row_indices[i] for i in defect_indices],
                       [right_x[i] for i in defect_indices],
                       c='yellow', s=30, marker='s', edgecolors='black', linewidths=1,
                       label='Right edge (DEFECT)', zorder=5)
            ax1.plot([row_indices[i] for i in defect_indices],
                    [avg_x[i] for i in defect_indices],
                    'yellow', linewidth=3, label='Average X (DEFECT)', zorder=4)
        
        ax1.set_xlabel('Row Index (Kernel Step)', fontsize=12)
        ax1.set_ylabel('X Coordinate (pixels)', fontsize=12)
        ax1.set_title('X-Coordinates over Rows (Yellow = Misalignment Defect)', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: X shift per row
        x_shifts = [0]  # First row has no shift
        for i in range(1, len(avg_x)):
            shift = avg_x[i] - avg_x[i-1]
            x_shifts.append(shift)
        
        # Normal shifts
        normal_shifts = [x_shifts[i] for i in normal_indices if i > 0]
        normal_rows = [row_indices[i] for i in normal_indices if i > 0]
        
        # Defect shifts
        defect_shifts = [x_shifts[i] for i in defect_indices]
        defect_rows = [row_indices[i] for i in defect_indices]
        
        if normal_rows:
            ax2.plot(normal_rows, normal_shifts, 'g-', linewidth=1, alpha=0.7, label='Normal shift')
        
        if defect_rows:
            ax2.plot(defect_rows, defect_shifts, 'yellow', linewidth=3, label='DEFECT shift', zorder=5)
            ax2.scatter(defect_rows, defect_shifts, c='yellow', s=50, marker='o',
                       edgecolors='black', linewidths=2, zorder=6)
        
        # Add threshold lines
        ax2.axhline(y=self.min_x_deviation, color='orange', linestyle='--',
                   label=f'Min deviation ({self.min_x_deviation}px)')
        ax2.axhline(y=-self.min_x_deviation, color='orange', linestyle='--')
        ax2.axhline(y=self.max_x_deviation, color='red', linestyle='--',
                   label=f'Max deviation ({self.max_x_deviation}px)')
        ax2.axhline(y=-self.max_x_deviation, color='red', linestyle='--')
        
        ax2.set_xlabel('Row Index (Kernel Step)', fontsize=12)
        ax2.set_ylabel('X Shift from Previous Row (pixels)', fontsize=12)
        ax2.set_title('X-Shift per Row (Yellow = Trend Defect)', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save graph
        graph_path = os.path.join(output_dir, f"{base_name}_x_trend_graph.png")
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"    Saved X-trend graph: {graph_path}")
        return graph_path
    
    def detect(self, image, image_path=None):
        """Run stripe misalignment detection using vertical line detector.

        Args:
            image: Input image (BGR or grayscale).
            image_path: Optional path for loading exclusion zones

        Returns:
            tuple: (visualization_bgr, defects)
        """
        # Use vertical line detector to get all edge points
        polygons_list, left_points, right_points, kernel_states, _ = \
            self.vertical_line_detector.detect_vertical_line(image, self.debug, image_path)
        
        if self.debug:
            print(f"\nAnalyzing {len(left_points)} left and {len(right_points)} right edge points for misalignment trends")
        
        # Analyze X-coordinate trends to find misalignment
        defects, trend_data = self.analyze_x_trends(left_points, right_points, self.debug)
        
        # Store trend data for graphing
        self._trend_data = trend_data
        
        # Create visualization
        visualization = self.create_visualization(image, defects, left_points, right_points, polygons_list)
        
        # Store debug images
        if self.debug and polygons_list:
            self._debug_polygon_vis = self.vertical_line_detector.create_visualization(
                image, polygons_list, left_points, right_points, kernel_states, []
            )
        
        return visualization, defects
    
    def create_visualization(self, original, defects, left_points, right_points, polygons_list):
        """Create visualization showing misalignment regions.

        Args:
            original: Original input image.
            defects: List of misalignment defect dicts.
            left_points: Left edge detection points.
            right_points: Right edge detection points.
            polygons_list: Polygon list from vertical line detector.

        Returns:
            BGR visualization image.
        """
        if len(original.shape) == 2:
            vis = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        else:
            vis = original.copy()
        
        overlay = vis.copy()
        
        # Sort points by Y for row-based highlighting
        left_sorted = sorted(left_points, key=lambda p: p['y'])
        right_sorted = sorted(right_points, key=lambda p: p['y'])
        
        # Draw misalignment regions - highlight the actual ROWS in YELLOW
        for defect in defects:
            y_start = defect['y_start']
            y_end = defect['y_end']
            
            # Get row indices for this defect
            if 'row_indices' in defect:
                for row_idx in defect['row_indices']:
                    if row_idx < len(left_sorted) and row_idx < len(right_sorted):
                        left_pt = left_sorted[row_idx]
                        right_pt = right_sorted[row_idx]
                        
                        y = left_pt['y']
                        left_x = left_pt['x']
                        right_x = right_pt['x']
                        
                        # Draw YELLOW horizontal bar across the line at this row
                        cv2.rectangle(overlay,
                                    (left_x - 50, y - 30),
                                    (right_x + 50, y + 30),
                                    (0, 255, 255),  # YELLOW
                                    -1)
            else:
                # Fallback: draw single rectangle
                x = defect['x']
                width_indicator = 100
                cv2.rectangle(overlay,
                            (x - width_indicator // 2, y_start),
                            (x + width_indicator // 2, y_end),
                            (0, 255, 255),  # YELLOW
                            -1)
        
        # Blend with original
        result = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
        
        return result
    
    def save_debug_images(self, output_dir, base_name):
        """Save debug images if available.

        Args:
            output_dir: Directory to save images.
            base_name: Base filename.

        Returns:
            List of saved paths or None.
        """
        debug_paths = []
        
        # Always save the graph
        if hasattr(self, '_trend_data') and self._trend_data is not None:
            graph_path = self.create_x_trend_graph(self._trend_data, output_dir, base_name)
            if graph_path:
                debug_paths.append(graph_path)
        
        if self.debug:
            if hasattr(self, '_debug_polygon_vis') and self._debug_polygon_vis is not None:
                path = os.path.join(output_dir, f"{base_name}_vertical_line_polygons.jpg")
                cv2.imwrite(path, self._debug_polygon_vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
                debug_paths.append(path)
                print(f"    Saved vertical line polygon debug: {path}")
        
        return debug_paths if debug_paths else None 