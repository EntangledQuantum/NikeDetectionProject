"""
Results Manager Module

This module handles saving results in various formats, generating reports,
and managing all output operations for the detection pipeline.

Author: Refactored Architecture
Date: 2024
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
from typing import List, Dict, Any
from enum import Enum

from .data_models import ImageResult, ProcessingConfig, TimingStats, to_dict


class ResultsSaver:
    """Handles saving of results in various formats"""
    
    @staticmethod
    def save_image_result(result: ImageResult, output_dir: str) -> None:
        """Save individual image result to JSON"""
        try:
            json_path = os.path.join(output_dir, f"{result.image_name}_results.json")
            result_dict = to_dict(result)
            with open(json_path, 'w') as f:
                json.dump(result_dict, f, indent=2, default=ResultsSaver._json_serializer)
        except Exception as e:
            print(f"    Warning: Could not save JSON results: {e}")
            # Save simplified version
            simplified_result = {
                'image_name': result.image_name,
                'processing_time': result.processing_time,
                'image_type': result.image_type.value,
                'detectors_used': result.detectors_used,
                'detection_counts': {name: detection.defect_count 
                                   for name, detection in result.detections.items()},
                'timing_summary': {name: detection.timing.get('total_time', 0) if detection.timing else 0
                                 for name, detection in result.detections.items()}
            }
            try:
                simplified_path = os.path.join(output_dir, f"{result.image_name}_results_simplified.json")
                with open(simplified_path, 'w') as f:
                    json.dump(simplified_result, f, indent=2)
                print(f"    Saved simplified results instead")
            except Exception as e2:
                print(f"    Could not save even simplified results: {e2}")
    
    @staticmethod
    def save_summary_report(results: List[ImageResult], output_dir: str, 
                          config: ProcessingConfig) -> None:
        """Save summary report with timing analysis"""
        # Calculate statistics
        stats = ResultsSaver._calculate_statistics(results)
        
        # Calculate timing statistics
        timing_stats = ResultsSaver._calculate_timing_statistics(results)
        
        # Save JSON summary
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'total_images': len(results),
            'detection_sensitivity': config.sensitivity,
            'optimization_settings': {
                'parallel_processing': config.enable_parallel_processing,
                'max_workers': config.max_workers,
                'individual_visualizations': config.save_individual_visualizations,
                'verbose_timing': config.verbose_timing
            },
            'image_type_distribution': stats['image_type_distribution'],
            'defect_statistics': stats['defect_statistics'],
            'processing_summary': stats['processing_summary'],
            'timing_analysis': timing_stats,
            'detailed_results': [to_dict(result) for result in results]
        }
        
        summary_path = os.path.join(output_dir, "defect_report.json")
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=ResultsSaver._json_serializer)
        
        print(f"📊 Summary report saved to: {summary_path}")
        
        # Print timing analysis
        ResultsSaver._print_timing_analysis(timing_stats)
        
        # Generate PDF report if requested
        if config.generate_report:
            ResultsSaver._generate_pdf_report(results, stats, output_dir)
    
    @staticmethod
    def _calculate_statistics(results: List[ImageResult]) -> Dict[str, Any]:
        """Calculate statistics from results"""
        # Image type distribution
        image_type_distribution = {}
        for result in results:
            image_type = result.image_type.value
            image_type_distribution[image_type] = image_type_distribution.get(image_type, 0) + 1
        
        # Defect statistics
        defect_statistics = {}
        processing_summary = {
            'successful_images': 0,
            'failed_images': 0,
            'total_processing_time': 0,
            'average_file_size_mb': 0.0
        }
        
        total_file_size = 0
        for result in results:
            if result.error:
                processing_summary['failed_images'] += 1
            else:
                processing_summary['successful_images'] += 1
            
            total_file_size += result.file_size_mb
            
            for detector_name, detection in result.detections.items():
                if detector_name not in defect_statistics:
                    defect_statistics[detector_name] = {
                        'total_defects': 0,
                        'affected_images': 0
                    }
                
                defect_statistics[detector_name]['total_defects'] += detection.defect_count
                if detection.defect_count > 0:
                    defect_statistics[detector_name]['affected_images'] += 1
        
        processing_summary['average_file_size_mb'] = float(total_file_size / len(results)) if results else 0.0
        
        return {
            'image_type_distribution': image_type_distribution,
            'defect_statistics': defect_statistics,
            'processing_summary': processing_summary
        }
    
    @staticmethod
    def _calculate_timing_statistics(results: List[ImageResult]) -> Dict[str, Any]:
        """Calculate detailed timing statistics"""
        timing_stats = {
            'total_images_processed': len(results),
            'detector_timing': {},
            'operation_timing': {},
            'optimization_impact': {}
        }
        
        detector_times = {}
        operation_times = {}
        
        for result in results:
            for detector_name, detection in result.detections.items():
                if detection.timing:
                    # Collect total times per detector
                    if detector_name not in detector_times:
                        detector_times[detector_name] = []
                    detector_times[detector_name].append(detection.timing.get('total_time', 0))
                    
                    # Collect operation times
                    for operation, op_time in detection.timing.items():
                        if operation != 'total_time':
                            if operation not in operation_times:
                                operation_times[operation] = []
                            operation_times[operation].append(op_time)
        
        # Calculate statistics for each detector
        for detector_name, times in detector_times.items():
            timing_stats['detector_timing'][detector_name] = {
                'average_time': float(np.mean(times)),
                'total_time': float(np.sum(times)),
                'min_time': float(np.min(times)),
                'max_time': float(np.max(times)),
                'std_time': float(np.std(times))
            }
        
        # Calculate statistics for each operation
        for operation, times in operation_times.items():
            timing_stats['operation_timing'][operation] = {
                'average_time': float(np.mean(times)),
                'total_time': float(np.sum(times)),
                'min_time': float(np.min(times)),
                'max_time': float(np.max(times))
            }
        
        # Calculate optimization impact
        shared_preprocessing_times = []
        for result in results:
            for detection in result.detections.values():
                if detection.timing and 'preprocessing_shared' in detection.timing:
                    shared_preprocessing_times.append(detection.timing['preprocessing_shared'])
        
        if shared_preprocessing_times:
            timing_stats['optimization_impact'] = {
                'shared_preprocessing_time_per_image': float(np.mean(shared_preprocessing_times)),
                'estimated_time_saved_per_image': float(np.mean(shared_preprocessing_times) * (len(detector_times) - 1))
            }
        
        return timing_stats
    
    @staticmethod
    def _print_timing_analysis(timing_stats: Dict[str, Any]) -> None:
        """Print detailed timing analysis to console"""
        print("\n⏱️  Detailed Timing Analysis:")
        print("=" * 60)
        
        # Detector timing
        if timing_stats.get('detector_timing'):
            print("\n📊 Detector Performance:")
            print("-" * 40)
            for detector_name, stats in timing_stats['detector_timing'].items():
                print(f"  {detector_name.title()}:")
                print(f"    Average: {stats['average_time']:.3f}s")
                print(f"    Total: {stats['total_time']:.3f}s")
                print(f"    Range: {stats['min_time']:.3f}s - {stats['max_time']:.3f}s")
        
        # Operation timing
        if timing_stats.get('operation_timing'):
            print("\n🔧 Operation Breakdown:")
            print("-" * 40)
            # Sort by total time
            sorted_ops = sorted(timing_stats['operation_timing'].items(), 
                              key=lambda x: x[1]['total_time'], reverse=True)
            
            for operation, stats in sorted_ops[:10]:  # Top 10 most time-consuming
                print(f"  {operation.replace('_', ' ').title()}:")
                print(f"    Average: {stats['average_time']:.3f}s")
                print(f"    Total: {stats['total_time']:.3f}s")
        
        # Optimization impact
        if timing_stats.get('optimization_impact'):
            print("\n🚀 Optimization Impact:")
            print("-" * 40)
            impact = timing_stats['optimization_impact']
            print(f"  Shared preprocessing time per image: {impact['shared_preprocessing_time_per_image']:.3f}s")
            print(f"  Estimated time saved per image: {impact['estimated_time_saved_per_image']:.3f}s")
            
            if impact['estimated_time_saved_per_image'] > 0:
                speedup = impact['estimated_time_saved_per_image'] / impact['shared_preprocessing_time_per_image']
                print(f"  Estimated speedup factor: {speedup:.2f}x")
    
    @staticmethod
    def _generate_pdf_report(results: List[ImageResult], stats: Dict[str, Any], 
                           output_dir: str) -> None:
        """Generate PDF report"""
        pdf_path = os.path.join(output_dir, "defect_detection_report.pdf")
        
        with PdfPages(pdf_path) as pdf:
            # Summary page
            fig, ax = plt.subplots(figsize=(8.5, 11))
            fig.suptitle("Optimized Defect Detection Report", fontsize=20, fontweight='bold')
            
            summary_text = f"Total Images Processed: {len(results)}\n\n"
            summary_text += "Image Type Distribution:\n" + "-" * 40 + "\n"
            
            for img_type, count in stats['image_type_distribution'].items():
                summary_text += f"  {img_type.title()}: {count} images\n"
            
            summary_text += "\nDefect Summary:\n" + "-" * 40 + "\n"
            
            for detector_name, detector_stats in stats['defect_statistics'].items():
                summary_text += f"\n{detector_name.replace('_', ' ').title()}:\n"
                summary_text += f"  Total Defects Found: {detector_stats['total_defects']}\n"
                summary_text += f"  Images with Defects: {detector_stats['affected_images']}\n"
                if detector_stats['affected_images'] > 0:
                    avg = detector_stats['total_defects'] / detector_stats['affected_images']
                    summary_text += f"  Average per Affected Image: {avg:.2f}\n"
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                   fontsize=12, verticalalignment='top', fontfamily='monospace')
            ax.axis('off')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        print(f"📄 PDF report saved to: {pdf_path}")
    
    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj) 