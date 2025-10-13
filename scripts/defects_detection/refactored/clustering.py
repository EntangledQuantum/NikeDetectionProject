"""
Defect Clustering Module

This module handles clustering of detected defects using sophisticated
algorithms and creates visualizations of clustered results.

Author: Refactored Architecture
Date: 2024
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

from .data_models import ClusterInfo


class DefectClusterer:
    """Clusters nearby defects of the same type using sophisticated algorithms"""
    
    def __init__(self, eps: float = 50.0, min_samples: int = 2, max_cluster_distance: float = 100.0):
        """
        Args:
            eps: Maximum distance between two samples for clustering (DBSCAN parameter)
            min_samples: Minimum number of samples in a neighborhood for a core point
            max_cluster_distance: Maximum distance to consider defects as clusterable
        """
        self.eps = eps
        self.min_samples = min_samples
        self.max_cluster_distance = max_cluster_distance
    
    def cluster_defects_by_type(self, defects: List[Dict[str, Any]]) -> Dict[str, List[ClusterInfo]]:
        """
        Cluster defects by type and proximity using DBSCAN clustering
        
        Args:
            defects: List of defect dictionaries with 'type', 'location', etc.
            
        Returns:
            Dictionary mapping defect types to lists of clustered defects
        """
        if not defects:
            return {}
        
        # Group defects by type first
        defects_by_type = {}
        for defect in defects:
            defect_type = defect.get('type', 'unknown')
            if defect_type not in defects_by_type:
                defects_by_type[defect_type] = []
            defects_by_type[defect_type].append(defect)
        
        # Cluster each type separately
        clustered_results = {}
        for defect_type, type_defects in defects_by_type.items():
            clustered_results[defect_type] = self._cluster_single_type(type_defects, defect_type)
        
        return clustered_results
    
    def _cluster_single_type(self, defects: List[Dict[str, Any]], defect_type: str) -> List[ClusterInfo]:
        """Cluster defects of a single type"""
        if len(defects) <= 1:
            # If only one defect, create a single-defect cluster
            if defects:
                return [ClusterInfo(
                    cluster_id=0,
                    defect_type=defect_type,
                    defects=defects,
                    centroid=defects[0]['location'],
                    bounding_box=self._calculate_bounding_box([defects[0]]),
                    cluster_size=1,
                    cluster_area=defects[0].get('area', 0)
                )]
            return []
        
        # Extract locations for clustering
        locations = []
        valid_defects = []
        
        for defect in defects:
            if 'location' in defect:
                locations.append(defect['location'])
                valid_defects.append(defect)
            elif 'bbox' in defect and len(defect['bbox']) >= 4:
                # Calculate centroid from bounding box
                minr, minc, maxr, maxc = defect['bbox'][:4]
                centroid = ((minr + maxr) // 2, (minc + maxc) // 2)
                locations.append(centroid)
                valid_defects.append(defect)
        
        if len(locations) < 2:
            # Not enough valid locations for clustering
            if valid_defects:
                return [ClusterInfo(
                    cluster_id=0,
                    defect_type=defect_type,
                    defects=valid_defects,
                    centroid=locations[0] if locations else (0, 0),
                    bounding_box=self._calculate_bounding_box(valid_defects),
                    cluster_size=len(valid_defects),
                    cluster_area=sum(d.get('area', 0) for d in valid_defects)
                )]
            return []
        
        # Apply DBSCAN clustering
        locations_array = np.array(locations)
        
        # Use adaptive eps based on data distribution
        adaptive_eps = min(self.eps, self._calculate_adaptive_eps(locations_array))
        
        clustering = DBSCAN(eps=adaptive_eps, min_samples=self.min_samples)
        cluster_labels = clustering.fit_predict(locations_array)
        
        # Group defects by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label == -1:  # Noise points get individual clusters
                label = f"noise_{i}"
            
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(valid_defects[i])
        
        # Create cluster objects
        result_clusters = []
        for cluster_id, cluster_defects in clusters.items():
            centroid = self._calculate_cluster_centroid(cluster_defects)
            bounding_box = self._calculate_bounding_box(cluster_defects)
            
            result_clusters.append(ClusterInfo(
                cluster_id=cluster_id,
                defect_type=defect_type,
                defects=cluster_defects,
                centroid=centroid,
                bounding_box=bounding_box,
                cluster_size=len(cluster_defects),
                cluster_area=sum(d.get('area', 0) for d in cluster_defects)
            ))
        
        return result_clusters
    
    def _calculate_adaptive_eps(self, locations: np.ndarray) -> float:
        """Calculate adaptive eps based on data distribution"""
        if len(locations) < 2:
            return self.eps
        
        # Calculate pairwise distances
        distances = cdist(locations, locations)
        
        # Use k-nearest neighbor distance (k=4) as adaptive eps
        k = min(4, len(locations) - 1)
        knn_distances = []
        for i in range(len(locations)):
            row_distances = distances[i]
            row_distances = np.sort(row_distances)[1:k+1]  # Exclude self (distance=0)
            knn_distances.append(np.mean(row_distances))
        
        # Use 75th percentile of k-NN distances
        adaptive_eps = float(np.percentile(knn_distances, 75))
        return min(adaptive_eps, self.max_cluster_distance)
    
    def _calculate_cluster_centroid(self, cluster_defects: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Calculate the centroid of a cluster"""
        x_coords = []
        y_coords = []
        
        for defect in cluster_defects:
            if 'location' in defect:
                x, y = defect['location']
                x_coords.append(x)
                y_coords.append(y)
            elif 'bbox' in defect and len(defect['bbox']) >= 4:
                minr, minc, maxr, maxc = defect['bbox'][:4]
                x_coords.append((minc + maxc) // 2)
                y_coords.append((minr + maxr) // 2)
        
        if x_coords and y_coords:
            return (int(np.mean(x_coords)), int(np.mean(y_coords)))
        return (0, 0)
    
    def _calculate_bounding_box(self, cluster_defects: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
        """Calculate the bounding box that encompasses all defects in cluster"""
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        
        for defect in cluster_defects:
            if 'bbox' in defect and len(defect['bbox']) >= 4:
                minr, minc, maxr, maxc = defect['bbox'][:4]
                min_x = min(min_x, minc)
                min_y = min(min_y, minr)
                max_x = max(max_x, maxc)
                max_y = max(max_y, maxr)
            elif 'location' in defect:
                x, y = defect['location']
                # Use a default size if no bbox available
                size = defect.get('area', 100) ** 0.5 / 2  # Approximate radius
                min_x = min(min_x, x - size)
                min_y = min(min_y, y - size)
                max_x = max(max_x, x + size)
                max_y = max(max_y, y + size)
        
        # Handle case where no valid coordinates found
        if min_x == float('inf'):
            return (0, 0, 10, 10)
        
        return (int(min_y), int(min_x), int(max_y), int(max_x))


class ClusterVisualizer:
    """Creates visualizations of clustered defects"""
    
    # Color scheme for different defect types
    TYPE_COLORS = {
        'debris': (0, 255, 255),      # Yellow
        'smudge': (255, 0, 255),      # Magenta
        'void': (0, 0, 255),          # Red
        'head_calibration': (255, 255, 0),  # Cyan
        'surface_treatment': (0, 255, 0),    # Green
        'unknown': (128, 128, 128)    # Gray
    }
    
    def create_cluster_visualization(self, image: np.ndarray, 
                                   clustered_defects: Dict[str, List[ClusterInfo]]) -> np.ndarray:
        """
        Create visualization with bright hollow circles for clustered defects
        
        Args:
            image: Original image
            clustered_defects: Dictionary of clustered defects by type
            
        Returns:
            Visualization image with clustered defects highlighted
        """
        vis = image.copy()
        
        cluster_id_counter = 0
        
        for defect_type, clusters in clustered_defects.items():
            color = self.TYPE_COLORS.get(defect_type, (255, 255, 255))  # White default
            
            for cluster in clusters:
                cluster_id_counter += 1
                centroid = cluster.centroid
                bounding_box = cluster.bounding_box
                cluster_size = cluster.cluster_size
                
                # Calculate circle radius based on cluster size and bounding box
                minr, minc, maxr, maxc = bounding_box
                bbox_width = maxc - minc
                bbox_height = maxr - minr
                base_radius = max(bbox_width, bbox_height) // 2 + 10
                
                # Scale radius based on cluster size
                size_multiplier = 1 + (cluster_size - 1) * 0.3  # Larger for more defects
                radius = int(base_radius * size_multiplier)
                radius = max(radius, 15)  # Minimum radius for visibility
                radius = min(radius, 100)  # Maximum radius to avoid huge circles
                
                # Draw bright hollow circle
                circle_thickness = max(3, radius // 10)  # Thickness scales with radius
                cv2.circle(vis, centroid, radius, color, circle_thickness)
                
                # Draw inner circle for better visibility
                inner_radius = max(5, radius // 3)
                cv2.circle(vis, centroid, inner_radius, color, 2)
                
                # Add cluster information text
                text = f"{defect_type[:4]}-{cluster_size}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                text_thickness = 1
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
                
                # Position text above the circle
                text_x = centroid[0] - text_width // 2
                text_y = centroid[1] - radius - 10
                
                # Ensure text stays within image bounds
                text_x = max(0, min(text_x, vis.shape[1] - text_width))
                text_y = max(text_height, min(text_y, vis.shape[0]))
                
                # Draw text background
                cv2.rectangle(vis, 
                            (text_x - 2, text_y - text_height - 2),
                            (text_x + text_width + 2, text_y + 2),
                            (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(vis, text, (text_x, text_y), font, font_scale, color, text_thickness)
                
                # Draw connecting lines between defects in the cluster if cluster has multiple defects
                if cluster_size > 1:
                    defects = cluster.defects
                    for defect in defects:
                        defect_location = defect.get('location')
                        if defect_location:
                            # Draw line from centroid to defect
                            cv2.line(vis, centroid, defect_location, color, 1)
                            # Mark individual defect with small circle
                            cv2.circle(vis, defect_location, 3, color, -1)
        
        return vis 