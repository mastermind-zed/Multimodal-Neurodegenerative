import zipfile
import os
import sys

def zip_data(source_dir, output_filename):
    print(f"Zipping {source_dir} to {output_filename}...")
    
    if not os.path.exists(source_dir):
        print(f"Error: Source directory {source_dir} does not exist.")
        return

    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Get total files for progress tracking
        file_list = []
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                file_list.append(os.path.join(root, file))
        
        total_files = len(file_list)
        print(f"Total files found: {total_files}")
        
        for i, file_path in enumerate(file_list):
            # Write file to zip with relative path
            zipf.write(file_path, os.path.relpath(file_path, source_dir))
            
            # Print periodic progress
            if (i + 1) % 500 == 0 or (i + 1) == total_files:
                percent = ((i + 1) / total_files) * 100
                print(f"Progress: {percent:.1f}% ({i + 1}/{total_files})")

    print(f"\nCompression complete! File size: {os.path.getsize(output_filename) / (1024*1024):.2f} MB")
    print(f"Now, upload '{output_filename}' to your Google Drive to use it in Colab.")

if __name__ == \"__main__\":
    # Default source for Alzheimer's processed data
    source = r\"d:\Machine Learning\Multimodal Neurodegenerative Research\data\processed_alzheimer\Data\"
    output = r\"d:\Machine Learning\Multimodal Neurodegenerative Research\alzheimer_data_colab.zip\"
    
    zip_data(source, output)
