"""
Annotation Validation Script
Validates YOLO format annotations before training
"""

import os
import yaml
from pathlib import Path
from loguru import logger
import argparse
from typing import List, Tuple
import cv2


class AnnotationValidator:
    def __init__(self, data_yaml: str):
        """
        Initialize annotation validator
        
        Args:
            data_yaml: Path to data.yaml configuration file
        """
        self.data_yaml = data_yaml
        self.config = self._load_config()
        self.errors = []
        self.warnings = []
        
    def _load_config(self) -> dict:
        """Load dataset configuration"""
        with open(self.data_yaml, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def validate(self) -> bool:
        """
        Run comprehensive validation
        
        Returns:
            True if validation passes, False otherwise
        """
        logger.info("Starting annotation validation...")
        
        # Validate config file
        self._validate_config()
        
        # Validate dataset splits
        for split in ['train', 'val']:
            if split in self.config:
                self._validate_split(split)
        
        # Print results
        self._print_results()
        
        return len(self.errors) == 0
    
    def _validate_config(self):
        """Validate data.yaml configuration"""
        logger.info("Validating configuration file...")
        
        # Check required fields
        required_fields = ['path', 'train', 'val', 'names', 'nc']
        for field in required_fields:
            if field not in self.config:
                self.errors.append(f"Missing required field in config: {field}")
        
        # Validate class count
        if 'nc' in self.config and 'names' in self.config:
            expected_classes = self.config['nc']
            actual_classes = len(self.config['names'])
            if expected_classes != actual_classes:
                self.errors.append(
                    f"Class count mismatch: nc={expected_classes} but {actual_classes} classes defined"
                )
    
    def _validate_split(self, split: str):
        """
        Validate a dataset split (train/val/test)
        
        Args:
            split: Split name ('train', 'val', 'test')
        """
        logger.info(f"Validating {split} split...")
        
        # Get paths
        base_path = Path(self.config['path'])
        images_path = base_path / self.config[split]
        labels_path = base_path / 'labels' / split
        
        # Check directories exist
        if not images_path.exists():
            self.errors.append(f"Images directory not found: {images_path}")
            return
        
        if not labels_path.exists():
            self.warnings.append(f"Labels directory not found: {labels_path}")
            return
        
        # Get image files
        image_files = self._get_image_files(images_path)
        logger.info(f"Found {len(image_files)} images in {split}")
        
        if len(image_files) == 0:
            self.warnings.append(f"No images found in {split} split")
            return
        
        # Validate each image-annotation pair
        missing_labels = 0
        invalid_annotations = 0
        
        for img_file in image_files:
            label_file = labels_path / f"{img_file.stem}.txt"
            
            # Check if label exists
            if not label_file.exists():
                missing_labels += 1
                continue
            
            # Validate annotation file
            if not self._validate_annotation_file(label_file, img_file):
                invalid_annotations += 1
        
        # Report statistics
        if missing_labels > 0:
            self.warnings.append(
                f"{split}: {missing_labels}/{len(image_files)} images missing labels"
            )
        
        if invalid_annotations > 0:
            self.errors.append(
                f"{split}: {invalid_annotations} annotation files have errors"
            )
        
        logger.info(f"{split} validation complete")
    
    def _get_image_files(self, directory: Path) -> List[Path]:
        """Get all image files from directory"""
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        return sorted(image_files)
    
    def _validate_annotation_file(
        self,
        label_file: Path,
        image_file: Path
    ) -> bool:
        """
        Validate single annotation file
        
        Args:
            label_file: Path to label file
            image_file: Path to corresponding image file
            
        Returns:
            True if valid, False otherwise
        """
        valid = True
        
        try:
            # Read image to get dimensions
            img = cv2.imread(str(image_file))
            if img is None:
                self.errors.append(f"Failed to read image: {image_file}")
                return False
            
            # Read annotations
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                
                # Check format
                if len(parts) < 5:
                    self.errors.append(
                        f"{label_file}:{i} - Invalid format (expected: class x y w h)"
                    )
                    valid = False
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Validate class ID
                    if class_id < 0 or class_id >= self.config['nc']:
                        self.errors.append(
                            f"{label_file}:{i} - Invalid class ID: {class_id}"
                        )
                        valid = False
                    
                    # Validate coordinates (should be normalized 0-1)
                    coords = [x_center, y_center, width, height]
                    for coord in coords:
                        if coord < 0 or coord > 1:
                            self.warnings.append(
                                f"{label_file}:{i} - Coordinate out of range [0,1]: {coord}"
                            )
                
                except ValueError as e:
                    self.errors.append(
                        f"{label_file}:{i} - Invalid number format: {e}"
                    )
                    valid = False
        
        except Exception as e:
            self.errors.append(f"Error validating {label_file}: {e}")
            valid = False
        
        return valid
    
    def _print_results(self):
        """Print validation results"""
        logger.info("\n" + "="*60)
        logger.info("VALIDATION RESULTS")
        logger.info("="*60)
        
        if len(self.errors) == 0 and len(self.warnings) == 0:
            logger.info("✓ All validations passed!")
        else:
            if len(self.errors) > 0:
                logger.error(f"\n✗ {len(self.errors)} ERRORS:")
                for error in self.errors[:10]:  # Show first 10
                    logger.error(f"  - {error}")
                if len(self.errors) > 10:
                    logger.error(f"  ... and {len(self.errors) - 10} more")
            
            if len(self.warnings) > 0:
                logger.warning(f"\n⚠ {len(self.warnings)} WARNINGS:")
                for warning in self.warnings[:10]:  # Show first 10
                    logger.warning(f"  - {warning}")
                if len(self.warnings) > 10:
                    logger.warning(f"  ... and {len(self.warnings) - 10} more")
        
        logger.info("="*60 + "\n")


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Validate YOLO annotations before training'
    )
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to data.yaml configuration file'
    )
    
    args = parser.parse_args()
    
    # Validate
    validator = AnnotationValidator(args.data)
    success = validator.validate()
    
    # Exit with appropriate code
    exit(0 if success else 1)


if __name__ == '__main__':
    main()