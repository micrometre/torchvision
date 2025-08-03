#!/usr/bin/env python3
"""
CLI tool to create COCO-format annotations.json from a directory of images.
Creates annotations structure optimized for license plate detection.
"""

import json
import os
import argparse
import random
from datetime import datetime
from PIL import Image
import glob
from PIL.ExifTags import TAGS, GPSTAGS

def get_exif_data(image):
    """Extract EXIF data from image if available"""
    exif_data = {}
    try:
        info = image._getexif()
        if info:
            for tag, value in info.items():
                decoded = TAGS.get(tag, tag)
                exif_data[decoded] = value
    except Exception:
        pass
    return exif_data

def get_gps_info(exif_data):
    """Extract GPS coordinates from EXIF data if available"""
    if not exif_data or 'GPSInfo' not in exif_data:
        return None
    
    gps_info = {}
    for key, value in exif_data['GPSInfo'].items():
        decoded = GPSTAGS.get(key, key)
        gps_info[decoded] = value
    
    # Check if we have the required GPS data
    if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
        lat = gps_info['GPSLatitude']
        lon = gps_info['GPSLongitude']
        lat_ref = gps_info.get('GPSLatitudeRef', 'N')
        lon_ref = gps_info.get('GPSLongitudeRef', 'E')
        
        # Convert coordinates to decimal degrees
        def convert_to_degrees(value):
            d, m, s = value
            return d + (m / 60.0) + (s / 3600.0)
        
        lat = convert_to_degrees(lat)
        if lat_ref == 'S':
            lat = -lat
            
        lon = convert_to_degrees(lon)
        if lon_ref == 'W':
            lon = -lon
            
        return {'latitude': lat, 'longitude': lon}
    
    return None

def get_capture_date(exif_data):
    """Extract capture date from EXIF data if available"""
    date_fields = ['DateTimeOriginal', 'DateTime', 'DateTimeDigitized']
    
    for field in date_fields:
        if field in exif_data:
            try:
                # Typical format: '2023:05:12 15:30:45'
                date_str = exif_data[field]
                date_obj = datetime.strptime(date_str, '%Y:%m:%d %H:%M:%S')
                return date_obj.isoformat()
            except Exception:
                pass
                
    return datetime.now().isoformat()

def get_image_info(image_path, image_id):
    """Extract image information for COCO format with enhanced metadata"""
    with Image.open(image_path) as img:
        width, height = img.size
        exif_data = get_exif_data(img)
        
    # Get GPS data if available
    gps_data = get_gps_info(exif_data)
    capture_date = get_capture_date(exif_data)
    
    image_info = {
        "id": image_id,
        "file_name": os.path.basename(image_path),
        "width": width,
        "height": height,
        "date_captured": capture_date,
        "license": 1,
        "coco_url": "",
        "flickr_url": ""
    }
    
    # Add GPS data if available
    if gps_data:
        image_info["geo_location"] = gps_data
        
    return image_info

def generate_dummy_license_plate_annotation(image_info, category_id=1, annotation_id=None):
    """Generate a dummy license plate annotation for the image"""
    width, height = image_info["width"], image_info["height"]
    
    # License plates are typically in the center-bottom area of the image
    # These are estimates and can be adjusted based on your specific needs
    plate_width = width * random.uniform(0.2, 0.4)  # 20-40% of image width
    plate_height = height * random.uniform(0.05, 0.15)  # 5-15% of image height
    
    # Position in the lower half of the image
    x = width * random.uniform(0.3, 0.7) - plate_width / 2
    y = height * (0.5 + random.uniform(0.1, 0.35)) - plate_height / 2
    
    # Ensure box is within image boundaries
    x = max(0, min(x, width - plate_width))
    y = max(0, min(y, height - plate_height))
    
    return {
        "id": annotation_id,
        "image_id": image_info["id"],
        "category_id": category_id,
        "bbox": [x, y, plate_width, plate_height],
        "area": plate_width * plate_height,
        "segmentation": [],
        "iscrowd": 0,
        "attributes": {"is_generated": True}
    }

def create_coco_structure(images_dir, categories=None, output_file="annotations.json", recursive=False, generate_dummy=False):
    """Create COCO format annotations.json from image directory"""
    
    # Default categories optimized for license plate detection
    if categories is None:
        categories = [
            {"id": 1, "name": "license_plate", "supercategory": "vehicle_part"},
            {"id": 2, "name": "vehicle", "supercategory": "vehicle"},
            {"id": 3, "name": "car", "supercategory": "vehicle"},
            {"id": 4, "name": "truck", "supercategory": "vehicle"},
            {"id": 5, "name": "motorcycle", "supercategory": "vehicle"}
        ]
    
    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "License Plate Detection Dataset",
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
    if recursive:
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(images_dir, '**', ext), recursive=True))
            image_files.extend(glob.glob(os.path.join(images_dir, '**', ext.upper()), recursive=True))
    else:
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(images_dir, ext)))
            image_files.extend(glob.glob(os.path.join(images_dir, ext.upper())))
    
    image_files.sort()
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Process each image
    annotation_id = 1
    for image_id, image_path in enumerate(image_files, 1):
        try:
            image_info = get_image_info(image_path, image_id)
            coco_data["images"].append(image_info)
            
            # Generate dummy license plate annotation if requested
            if generate_dummy:
                annotation = generate_dummy_license_plate_annotation(
                    image_info, 
                    category_id=1,  # license_plate category
                    annotation_id=annotation_id
                )
                coco_data["annotations"].append(annotation)
                annotation_id += 1
                
            print(f"Processed: {os.path.basename(image_path)}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    # Save annotations file
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"\nCreated {output_file} with {len(coco_data['images'])} images")
    print(f"Categories: {[cat['name'] for cat in categories]}")
    
    if generate_dummy:
        print(f"Generated {len(coco_data['annotations'])} dummy license plate annotations")
        print("\nNote: Dummy annotations are approximate and should be refined for accurate detection.")
    else:
        print("\nNote: This creates an empty annotations structure.")
        print("You'll need to add actual bounding box annotations manually or with an annotation tool.")

def main():
    parser = argparse.ArgumentParser(
        description="Create COCO-format annotations.json for license plate detection"
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
        help="Category names (e.g., --categories license_plate car truck)"
    )
    parser.add_argument(
        "--list-images",
        action="store_true",
        help="Just list found images without creating annotations"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search for images recursively in subdirectories"
    )
    parser.add_argument(
        "--generate-dummy",
        action="store_true",
        help="Generate dummy license plate annotations (useful as starting point)"
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
        
        if args.recursive:
            for ext in extensions:
                image_files.extend(glob.glob(os.path.join(args.images_dir, '**', ext), recursive=True))
                image_files.extend(glob.glob(os.path.join(args.images_dir, '**', ext.upper()), recursive=True))
        else:
            for ext in extensions:
                image_files.extend(glob.glob(os.path.join(args.images_dir, ext)))
                image_files.extend(glob.glob(os.path.join(args.images_dir, ext.upper())))
        
        image_files.sort()
        print(f"Found {len(image_files)} images:")
        for img in image_files:
            print(f"  {os.path.relpath(img, args.images_dir)}")
    else:
        # Create annotations
        create_coco_structure(
            args.images_dir, 
            categories, 
            args.output, 
            recursive=args.recursive,
            generate_dummy=args.generate_dummy
        )
    
    return 0

if __name__ == "__main__":
    exit(main())