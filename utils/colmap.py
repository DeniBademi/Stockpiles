import os
import subprocess
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import math
import laspy
from .point_cloud import convert_ply_to_las

def extract_gps_metadata(image_path):
    """
    Extract GPS metadata from an image.

    Args:
        image_path: str
            Path to the image file

    Returns:
        tuple
            (latitude, longitude, altitude) in degrees and meters
    """
    try:
        image = Image.open(image_path)
        exif = image._getexif()

        if exif is None:
            return None

        # GPS tags
        gps_latitude = None
        gps_longitude = None
        gps_altitude = None

        for tag_id in exif:
            tag = TAGS.get(tag_id, tag_id)
            data = exif[tag_id]

            if tag == 'GPSInfo':
                for key in data.keys():
                    if key == 2:  # Latitude
                        gps_latitude = data[key]
                    elif key == 4:  # Longitude
                        gps_longitude = data[key]
                    elif key == 6:  # Altitude
                        gps_altitude = data[key]

        if gps_latitude and gps_longitude:
            # Convert to decimal degrees
            lat = gps_latitude[0] + gps_latitude[1]/60 + gps_latitude[2]/3600
            lon = gps_longitude[0] + gps_longitude[1]/60 + gps_longitude[2]/3600

            # Convert altitude to meters
            alt = float(gps_altitude) if gps_altitude else 0.0

            return (lat, lon, alt)

    except Exception as e:
        print(f"Error extracting GPS metadata: {e}")
        return None

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two GPS coordinates in meters.

    Args:
        lat1, lon1: float
            First point coordinates in degrees
        lat2, lon2: float
            Second point coordinates in degrees

    Returns:
        float
            Distance in meters
    """
    R = 6371000  # Earth's radius in meters

    # Convert to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    distance = R * c

    return distance

def run_colmap(image_dir, output_dir):
    """
    Run COLMAP reconstruction on a directory of images.

    Args:
        image_dir: str
            Directory containing input images
        output_dir: str
            Directory for COLMAP output files

    Returns:
        bool
            True if successful, False otherwise
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Run COLMAP feature extraction
        subprocess.run([
            'colmap', 'feature_extractor',
            '--database_path', os.path.join(output_dir, 'database.db'),
            '--image_path', image_dir
        ], check=True)

        # Run COLMAP exhaustive matcher
        subprocess.run([
            'colmap', 'exhaustive_matcher',
            '--database_path', os.path.join(output_dir, 'database.db')
        ], check=True)

        # Run COLMAP mapper
        subprocess.run([
            'colmap', 'mapper',
            '--database_path', os.path.join(output_dir, 'database.db'),
            '--image_path', image_dir,
            '--output_path', output_dir
        ], check=True)

        return True

    except subprocess.CalledProcessError as e:
        print(f"Error running COLMAP: {e}")
        return False

def scale_ply_with_gps(ply_path, image_dir, output_las_path):
    """
    Scale a PLY file using GPS metadata from images.

    Args:
        ply_path: str
            Path to input PLY file
        image_dir: str
            Directory containing images with GPS metadata
        output_las_path: str
            Path for output LAS file

    Returns:
        bool
            True if successful, False otherwise
    """
    try:
        # Get all images in directory
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if len(image_files) < 2:
            print("Need at least 2 images with GPS metadata")
            return False

        # Extract GPS coordinates from first two images
        gps1 = extract_gps_metadata(os.path.join(image_dir, image_files[0]))
        gps2 = extract_gps_metadata(os.path.join(image_dir, image_files[1]))

        if not gps1 or not gps2:
            print("Could not extract GPS metadata from images")
            return False

        # Calculate real-world distance
        real_distance = calculate_distance(gps1[0], gps1[1], gps2[0], gps2[1])

        # Convert PLY to LAS with scaling
        success = convert_ply_to_las(ply_path, output_las_path, real_distance)

        return success

    except Exception as e:
        print(f"Error scaling PLY with GPS: {e}")
        return False