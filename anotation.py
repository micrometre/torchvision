#!/usr/bin/env python3
"""
CLI tool to create COCO-format annotations.json from a directory of images.
Creates empty annotations structure that can be populated later.
"""

import json
import os
import argparse
from datetime import datetime
from PIL import Image
import glob

def get_image_info(image_path, image_id):
    """Extract image information for COCO format"""
    with Image.open(image_path) as img:
        width, height = img.size
    
    return {
        "id": image_id,
        "file_name": os.path.basename(image_path),
        "width": width,
        "height": height,
        "date_captured": datetime.now().isoformat(),
        "license": 1,
        "coco_url": "",
        "flickr_url": ""
    }

def create_coco_structure(images_dir, categories=None, output_file="annotations.json"):
    """Create COCO format annotations.json from image directory"""
    
    # Default categories if none provided
    if categories is None:
        categories = [
            {"id": 1, "name": "person", "supercategory": "person"},
            {"id": 2, "name": "car", "supercategory": "vehicle"},
            {"id": 3, "name": "bicycle", "supercategory": "vehicle"}
        ]
    
    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "Generated annotations",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Auto-generated",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    
    # Find all images
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
        image_files.extend(glob.glob(os.path.join(images_dir, ext.upper())))
    
    image_files.sort()
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Process each image
    for image_id, image_path in enumerate(image_files, 1):
        try:
            image_info = get_image_info(image_path, image_id)
            coco_data["images"].append(image_info)
            print(f"Processed: {os.path.basename(image_path)}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Save annotations file
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\nCreated {output_file} with {len(coco_data['images'])} images")
    print(f"Categories: {[cat['name'] for cat in categories]}")
    print("\nNote: This creates an empty annotations structure.")
    print("You'll need to add actual bounding box annotations manually or with an annotation tool.")

def main():
    parser = argparse.ArgumentParser(
        description="Create COCO-format annotations.json from image directory"
    )
    parser.add_argument(
        "images_dir", 
        help="Directory containing images"
    )
    parser.add_argument(
        "-o", "--output", 
        default="annotations.json",
        help="Output annotations file (default: annotations.json)"
    )
    parser.add_argument(
        "-c", "--categories",
        nargs="+",
        help="Category names (e.g., --categories person car bicycle)"
    )
    parser.add_argument(
        "--list-images",
        action="store_true",
        help="Just list found images without creating annotations"
    )
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.images_dir):
        print(f"Error: {args.images_dir} is not a valid directory")
        return 1
    
    # Create categories from names if provided
    categories = None
    if args.categories:
        categories = [
            {"id": i+1, "name": name, "supercategory": "object"}
            for i, name in enumerate(args.categories)
        ]
    
    if args.list_images:
        # Just list images
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(args.images_dir, ext)))
            image_files.extend(glob.glob(os.path.join(args.images_dir, ext.upper())))
        
        image_files.sort()
        print(f"Found {len(image_files)} images:")
        for img in image_files:
            print(f"  {os.path.basename(img)}")
    else:
        # Create annotations
        create_coco_structure(args.images_dir, categories, args.output)
    
    return 0

if __name__ == "__main__":
    exit(main())