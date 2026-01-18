import requests
import re
import os
import subprocess
from bs4 import BeautifulSoup

# DWD GRIB definitions base URL
DWD_GRIB_URL = "https://opendata.dwd.de/weather/lib/grib/"

def get_latest_grib_definitions():
    """Fetches the latest GRIB definitions from DWD's open data server."""
    response = requests.get(DWD_GRIB_URL)
    if response.status_code != 200:
        print(f"Error: Unable to access {DWD_GRIB_URL}")
        return None

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract available GRIB definition file names
    files = []
    for link in soup.find_all("a"):
        filename = link.get("href")
        if filename and filename.startswith("eccodes_definitions.edzw") and filename.endswith(".tar.bz2"):
            files.append(filename)

    if not files:
        print("No GRIB definition files found!")
        return None

    # Sort files based on version numbers
    files.sort(reverse=True, key=lambda x: list(map(int, re.findall(r"\d+", x))))
    
    # Select the latest file
    latest_file = files[0]
    print(f"Latest GRIB definitions file found: {latest_file}")

    return latest_file

def download_file(filename):
    """Downloads the given file from DWD's GRIB repository."""
    url = DWD_GRIB_URL + filename
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        print(f"Downloaded: {filename}")
        return filename
    else:
        print(f"Error: Unable to download {filename}")
        return None

def extract_and_set_path(filename, target_directory=os.path.expanduser("~/eccodes_dwd")):
    """Extracts the .tar.bz2 file into ~/eccodes_dwd and sets ECCODES_DEFINITION_PATH."""
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    print(f"Extracting {filename} to {target_directory}...")
    subprocess.run(["tar", "-xvjf", filename, "-C", target_directory], check=True)

    # Locate the extracted definitions path
    extracted_subdirs = [os.path.join(target_directory, d) for d in os.listdir(target_directory) if os.path.isdir(os.path.join(target_directory, d))]
    
    definitions_path = None
    for subdir in extracted_subdirs:
        if "definitions" in os.listdir(subdir):  # Check if a 'definitions' folder exists
            definitions_path = os.path.join(subdir, "definitions")
            break

    if definitions_path:
        os.environ["ECCODES_DEFINITION_PATH"] = f"{definitions_path}:{os.environ.get('ECCODES_DEFINITION_PATH', '')}"
        print(f"ECCODES_DEFINITION_PATH set to: {os.environ['ECCODES_DEFINITION_PATH']}")

        # Make this change permanent
        with open(os.path.expanduser("~/.bashrc"), "a") as bashrc:
            bashrc.write(f'\nexport ECCODES_DEFINITION_PATH={definitions_path}:$ECCODES_DEFINITION_PATH\n')

        print("ECCODES_DEFINITION_PATH has been added to ~/.bashrc. Run `source ~/.bashrc` to apply changes.")
        return definitions_path
    else:
        print("Error: Could not find a 'definitions' folder in the extracted archive.")
        return None

if __name__ == "__main__":
    latest_grib_def = get_latest_grib_definitions()
    if latest_grib_def:
        downloaded_file = download_file(latest_grib_def)
        if downloaded_file:
            extract_and_set_path(downloaded_file)
