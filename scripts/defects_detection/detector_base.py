"""
Base Detector Class
Provides standard interface for all defect detectors

Author: Assistant
Date: 2024
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path


class BaseDetector:
    """Base class for all defect detectors"""
    
    def __init__(self):
        """Initialize base detector with exclusion zone support"""
        self.exclusion_zones = []
    
    def detect(self, image, image_path=None):
        """
        Detect defects in the image
        
        Args:
            image: Input image (BGR or grayscale)
            image_path: Optional path to the image file (for loading exclusion zones)
            
        Returns:
            tuple: (visualization_image, defect_list)
        """
        raise NotImplementedError("Subclasses must implement detect method")
    
    def load_exclusion_zones(self, image_path):
        """Load exclusion zones from JSON file with same name as image"""
        try:
            # Get JSON file path (same name as image, different extension)
            image_path = Path(image_path)
            json_path = image_path.with_suffix('.json')
            
            if not json_path.exists():
                self.exclusion_zones = []
                return
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract exclusion zones from JSON
            self.exclusion_zones = []
            if 'exclusion_zones' in data:
                for zone in data['exclusion_zones']:
                    bbox = zone.get('bounding_box_pixels', {})
                    
                    # Convert coordinates like tiff_extractor.py does:
                    # Convert to positive values if negative and ensure proper ordering
                    raw_top_x = float(bbox.get('top_x', 0))
                    raw_top_y = float(bbox.get('top_y', 0))
                    raw_bottom_x = float(bbox.get('bottom_x', 0))
                    raw_bottom_y = float(bbox.get('bottom_y', 0))
                    
                    x1 = int(min(abs(raw_top_x), abs(raw_bottom_x)))
                    y1 = int(min(abs(raw_top_y), abs(raw_bottom_y)))
                    x2 = int(max(abs(raw_top_x), abs(raw_bottom_x)))
                    y2 = int(max(abs(raw_top_y), abs(raw_bottom_y)))
                    
                    self.exclusion_zones.append({
                        'top_x': x1,
                        'top_y': y1,
                        'bottom_x': x2,
                        'bottom_y': y2,
                        'name': zone.get('name', 'unnamed')
                    })
            
            if self.exclusion_zones and hasattr(self, 'debug') and getattr(self, 'debug', False):
                print(f"Loaded {len(self.exclusion_zones)} exclusion zones from {json_path}")
                for i, zone in enumerate(self.exclusion_zones):
                    print(f"  Zone {i+1} '{zone['name']}': ({zone['top_x']}, {zone['top_y']}) to ({zone['bottom_x']}, {zone['bottom_y']})")
            
        except Exception as e:
            print(f"Warning: Could not load exclusion zones from {json_path}: {e}")
            self.exclusion_zones = []
    
    def is_point_in_exclusion_zone(self, x, y):
        """Check if a point falls within any exclusion zone"""
        for zone in self.exclusion_zones:
            zone_x1 = min(zone['top_x'], zone['bottom_x'])
            zone_y1 = min(zone['top_y'], zone['bottom_y'])
            zone_x2 = max(zone['top_x'], zone['bottom_x'])
            zone_y2 = max(zone['top_y'], zone['bottom_y'])
            
            if zone_x1 <= x <= zone_x2 and zone_y1 <= y <= zone_y2:
                return True, zone
        
        return False, None
    
    def is_region_in_exclusion_zone(self, x, y, width, height):
        """Check if a rectangular region overlaps with any exclusion zone"""
        for zone in self.exclusion_zones:
            zone_x1 = min(zone['top_x'], zone['bottom_x'])
            zone_y1 = min(zone['top_y'], zone['bottom_y'])
            zone_x2 = max(zone['top_x'], zone['bottom_x'])
            zone_y2 = max(zone['top_y'], zone['bottom_y'])
            
            # Check for overlap
            if (x < zone_x2 and x + width > zone_x1 and
                y < zone_y2 and y + height > zone_y1):
                return True, zone
        
        return False, None
    
    def draw_exclusion_zones(self, image):
        """Draw exclusion zones on the image for visualization"""
        if not self.exclusion_zones:
            return image
        
        overlay = image.copy()
        
        for i, zone in enumerate(self.exclusion_zones):
            # Convert zone coordinates to proper order
            x1 = min(zone['top_x'], zone['bottom_x'])
            y1 = min(zone['top_y'], zone['bottom_y'])
            x2 = max(zone['top_x'], zone['bottom_x'])
            y2 = max(zone['top_y'], zone['bottom_y'])
            
            # Draw exclusion zone as magenta rectangle
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 3)
            
            # Add zone label
            label = f"Exclusion {i+1}: {zone.get('name', 'unnamed')}"
            cv2.putText(overlay, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        
        return overlay
    
    def _standardize_output(self, result):
        """
        Standardize detector output to consistent format
        
        Args:
            result: Raw detector output (dict or other format)
            
        Returns:
            tuple: (visualization_image, defect_list)
        """
        if isinstance(result, dict):
            visualization = result.get('visualization')
            defects = result.get('defects', [])
            return visualization, defects
        elif isinstance(result, tuple) and len(result) == 2:
            return result
        else:
            raise ValueError(f"Unexpected detector output format: {type(result)}")
    
    def detect_wrapper(self, image):
        """
        Wrapper that ensures consistent output format
        
        Args:
            image: Input image
            
        Returns:
            tuple: (visualization_image, defect_list)
        """
        result = self._detect_impl(image)
        return self._standardize_output(result)
    
    def _detect_impl(self, image):
        """Implementation of detection logic - to be overridden by subclasses"""
        raise NotImplementedError("Subclasses must implement _detect_impl method") 