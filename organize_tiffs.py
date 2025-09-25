
import os
import glob
import rasterio
import shutil

def organize_tiffs_by_crs(input_dir):
    """
    Organizes TIFF files in a directory by their CRS.

    Args:
        input_dir (str): The path to the directory containing TIFF files.
    """
    tiff_files = glob.glob(os.path.join(input_dir, '*.tiff'))

    for tiff_file in tiff_files:
        try:
            with rasterio.open(tiff_file) as src:
                epsg_code = src.crs.to_epsg()
                if epsg_code:
                    dest_dir = os.path.join(input_dir, f'epsg_{epsg_code}')
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.move(tiff_file, os.path.join(dest_dir, os.path.basename(tiff_file)))
                    print(f"Moved {os.path.basename(tiff_file)} to {dest_dir}")
        except Exception as e:
            print(f"Could not process {os.path.basename(tiff_file)}: {e}")

if __name__ == '__main__':
    organize_tiffs_by_crs('/storage/skyterra/kz')
