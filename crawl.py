import requests
import os
from duckduckgo_search import DDGS
import time
from requests.exceptions import RequestException

def download_images(query, save_folder, prefix, max_results=300):
    """Helper function to download images for a specific query"""
    ddgs = DDGS()
    print(f"Searching for {query}...")
    
    try:
        # Get extra results to account for potential failures
        results = list(ddgs.images(query, max_results=max_results + 20))
    except Exception as e:
        print(f"Error getting search results for {query}: {e}")
        return
    
    print(f"Found {len(results)} images for {query}")
    
    successful_downloads = 0
    result_index = 0
    
    while successful_downloads < max_results and result_index < len(results):
        img_path = os.path.join(save_folder, f"{prefix}_{successful_downloads}.jpg")
        
        # Skip if file already exists
        if os.path.exists(img_path):
            print(f"Skipping {prefix}_{successful_downloads}.jpg - already exists")
            successful_downloads += 1
            continue
            
        try:
            img_url = results[result_index]['image']
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()
            
            with open(img_path, "wb") as f:
                f.write(response.content)
            
            print(f"Downloaded {prefix}_{successful_downloads}.jpg")
            successful_downloads += 1
            time.sleep(1)  # Add delay between downloads
            
        except RequestException as e:
            print(f"Network error downloading image {result_index} for {query}: {e}")
        except Exception as e:
            print(f"Error downloading image {result_index} for {query}: {e}")
        
        result_index += 1
    
    if successful_downloads < max_results:
        print(f"Warning: Only downloaded {successful_downloads}/{max_results} images for {query}")
    else:
        print(f"Successfully downloaded {max_results} images for {query}")
    
    time.sleep(2)  # Add delay between queries

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
