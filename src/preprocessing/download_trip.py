import os
import urllib.request
import urllib.parse

BASE_LFS_URL = "https://media.githubusercontent.com/media/onyekpeu/IO-VNBD/master/"

# Selected multi-driver trip list for rich dataset diversity and generalization
FEATURED_TRIPS = [
    ("Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S1", "S-S1.csv", "V-S1.csv"),
    ("Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S2", "S-S2.csv", "V-S2.csv"),
    ("Synchronised V abd S datasets/Categorised IOVNB Dataset/S (Driver A)/S4", "S-S4.csv", "V-S4.csv"),
    ("Synchronised V abd S datasets/Categorised IOVNB Dataset/M (Driver B)", "S-M.csv", "V-M.csv"),
    ("Synchronised V abd S datasets/Categorised IOVNB Dataset/Y (Driver D)/Y1", "S-Y1.csv", "V-Y1.csv"),
    ("Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw02", "S-Vw2.csv", "V-Vw2.csv"),
    ("Synchronised V abd S datasets/Categorised IOVNB Dataset/Vw (Driver E)/Vw03", "S-Vw3.csv", "V-Vw3.csv"),
    ("Synchronised V abd S datasets/Categorised IOVNB Dataset/Vta (Driver E)/Vta01b", "S-Vta1b.csv", "V-Vta1b.csv"),
    ("Synchronised V abd S datasets/Categorised IOVNB Dataset/Vta (Driver E)/Vta06", "S-Vta6.csv", "V-vta6.csv"),
    ("Synchronised V abd S datasets/Categorised IOVNB Dataset/Vtb (Driver E)/Vtb01", "S-Vtb1.csv", "V-vtb1.csv"),
]

def download_file(relative_path, target_filepath):
    parts = relative_path.replace("\\", "/").split("/")
    encoded_parts = [urllib.parse.quote(part) for part in parts]
    full_url = BASE_LFS_URL + "/".join(encoded_parts)
    
    if os.path.exists(target_filepath) and os.path.getsize(target_filepath) > 1000:
        print(f"Skipping (already downloaded): {relative_path}")
        return
        
    print(f"Downloading: {relative_path}")
    os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            print(f"\rProgress: {downloaded / 1024 / 1024:.2f} MB / {total_size / 1024 / 1024:.2f} MB ({percent:.1f}%)", end="")
        else:
            print(f"\rDownloaded: {downloaded / 1024 / 1024:.2f} MB", end="")
            
    try:
        urllib.request.urlretrieve(full_url, target_filepath, reporthook=report_progress)
        print("\nDownload complete!\n")
    except Exception as e:
        print(f"\nWarning: Could not download {full_url}: {e}\n")

def download_all_featured_trips(base_target="data/raw/IO-VNBD/IO-VNBD-master"):
    print(f"--- Downloading {len(FEATURED_TRIPS)} Multi-Driver Trips from IO-VNBD ---")
    for trip_rel_dir, s_file, v_file in FEATURED_TRIPS:
        s_rel = f"{trip_rel_dir}/{s_file}"
        v_rel = f"{trip_rel_dir}/{v_file}"
        
        s_target = os.path.join(base_target, s_rel)
        v_target = os.path.join(base_target, v_rel)
        
        download_file(s_rel, s_target)
        download_file(v_rel, v_target)

if __name__ == "__main__":
    download_all_featured_trips()
