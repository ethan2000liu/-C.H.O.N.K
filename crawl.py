import requests
import os
import time
import threading
from duckduckgo_search import DDGS
from requests.exceptions import RequestException
from concurrent.futures import ThreadPoolExecutor

def download_images(query, save_folder, prefix, max_results=300, num_threads=50):
    """Download images for a specific query using threads while keeping sequential filenames."""
    ddgs = DDGS()
    print(f"Searching for {query}...")

    try:
        # Get extra results to account for potential failures
        results = list(ddgs.images(query, max_results=max_results + 20))
        print(f"Found {len(results)} images for {query}")
    except Exception as e:
        print(f"Error getting search results for {query}: {e}")
        return

    # Shared counters for successes and current result index
    successful_downloads = 0
    result_index = 0
    lock = threading.Lock()

    def worker():
        nonlocal successful_downloads, result_index
        while True:
            # Get the next result safely
            with lock:
                if successful_downloads >= max_results or result_index >= len(results):
                    return  # We're done
                current_result_index = result_index
                result_index += 1

            # Attempt to download the image outside the lock
            try:
                img_url = results[current_result_index]['image']
                response = requests.get(img_url, timeout=10)
                response.raise_for_status()
                image_data = response.content
            except RequestException as e:
                print(f"Network error downloading image {current_result_index} for {query}: {e}")
                continue
            except Exception as e:
                print(f"Error downloading image {current_result_index} for {query}: {e}")
                continue

            # Assign a sequential file index for a successful download
            with lock:
                file_index = successful_downloads
                successful_downloads += 1

            img_path = os.path.join(save_folder, f"{prefix}_{file_index}.jpg")

            # If the file exists already, skip saving
            if os.path.exists(img_path):
                print(f"Skipping {prefix}_{file_index}.jpg - already exists")
                continue

            try:
                with open(img_path, "wb") as f:
                    f.write(image_data)
                print(f"Downloaded {prefix}_{file_index}.jpg")
            except Exception as e:
                print(f"Error saving image {current_result_index} for {query}: {e}")
            time.sleep(1)  # Add delay between downloads for each thread

    # Launch worker threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker) for _ in range(num_threads)]
        # Wait for all workers to complete
        for future in futures:
            future.result()

    if successful_downloads < max_results:
        print(f"Warning: Only downloaded {successful_downloads}/{max_results} images for {query}")
    else:
        print(f"Successfully downloaded {max_results} images for {query}")

    time.sleep(1)  # Delay between queries

# Create main folders
normal_folder = "normal_cat"
chonky_folder = "chonky_cat"
os.makedirs(normal_folder, exist_ok=True)
os.makedirs(chonky_folder, exist_ok=True)

# Download normal cats (more variety in queries)
print("Starting normal cats download...")
download_images("normal cat", normal_folder, "normal_cat", max_results=300)
download_images("cat", normal_folder, "cat", max_results=300)
download_images("small cat", normal_folder, "small_cat", max_results=300)
download_images("skinny cat", normal_folder, "skinny_cat", max_results=300)
download_images("average cat", normal_folder, "average_cat", max_results=300)

# Download chonky cats (3 different queries)
print("Starting chonky cats download...")
download_images("fat cat", chonky_folder, "fat_cat", max_results=300)
download_images("chonky cat", chonky_folder, "chonky_cat", max_results=300)
download_images("obese cat", chonky_folder, "obese_cat", max_results=300)

print("All downloads completed!")
