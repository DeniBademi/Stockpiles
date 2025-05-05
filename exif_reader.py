import os
import exifread
import csv
import utm

def dms_to_decimal(degrees, minutes, seconds):
    return degrees + (minutes / 60.0) + (seconds / 3600.0)

def parse_gps_value(gps_value):
    d = float(gps_value[0])
    m = float(gps_value[1])
    s = float(gps_value[2].num) / float(gps_value[2].den)
    return dms_to_decimal(d, m, s)

def extract_gps_from_images(image_folder, output_csv="gps_data.csv"):
    data = []

    for fname in sorted(os.listdir(image_folder)):
        if not fname.lower().endswith((".jpg", ".jpeg")):
            continue

        path = os.path.join(image_folder, fname)
        with open(path, 'rb') as f:
            tags = exifread.process_file(f, details=False)

        try:
            lat_ref = tags["GPS GPSLatitudeRef"].printable
            lat = parse_gps_value(tags["GPS GPSLatitude"].values)
            if lat_ref == "S":
                lat = -lat

            lon_ref = tags["GPS GPSLongitudeRef"].printable
            lon = parse_gps_value(tags["GPS GPSLongitude"].values)
            if lon_ref == "W":
                lon = -lon

            alt = float(tags["GPS GPSAltitude"].printable)

            # Convert lat/lon to UTM (meters)
            easting, northing, zone, _ = utm.from_latlon(lat, lon)

            data.append((fname, easting, northing, alt))

        except KeyError:
            print(f"GPS data not found for image: {fname}")

    # Save to CSV
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image", "easting", "northing", "altitude"])
        writer.writerows(data)

    print(f"Extracted GPS data saved to: {output_csv}")

if __name__ == "__main__":
    image_folder = "data/flight_1/images"  # 👈 change to your image directory if needed
    extract_gps_from_images(image_folder)