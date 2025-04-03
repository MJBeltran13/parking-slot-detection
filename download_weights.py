import requests
import os

def download_weights():
    url = 'https://pjreddie.com/media/files/yolov3.weights'
    save_path = 'model_data/yolov3.weights'
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Download the weights
    print('Downloading YOLOv3 weights...')
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f:
        if total_size == 0:
            f.write(response.content)
        else:
            downloaded = 0
            total_size_mb = total_size / (1024 * 1024)
            for data in response.iter_content(chunk_size=8192):
                downloaded += len(data)
                f.write(data)
                done = int(50 * downloaded / total_size)
                print(f'\r[{"=" * done}{"." * (50-done)}] {downloaded/(1024*1024):.1f}/{total_size_mb:.1f} MB', end='')
    
    print('\nDownload complete!')

if __name__ == '__main__':
    download_weights() 