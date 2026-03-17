import shutil
import os
import time

def zip_directory(source_dir, output_filename):
    print(f"Zipping {source_dir} to {output_filename}...")
    start_time = time.time()
    # base_name is the path to the zip file without extension
    base_name = os.path.splitext(output_filename)[0]
    shutil.make_archive(base_name, 'zip', source_dir)
    end_time = time.time()
    print(f"Finished in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    # Alzheimer's
    alz_source = r"d:\Machine Learning\Multimodal Neurodegenerative Research\data\alzheimer\Data"
    alz_out = r"d:\Machine Learning\Multimodal Neurodegenerative Research\data\alzheimer_data.zip"
    
    # Remove partial zip if exists
    if os.path.exists(alz_out):
        os.remove(alz_out)
        
    zip_directory(alz_source, alz_out)
    
    # Parkinson's
    pk_source = r"d:\Machine Learning\Multimodal Neurodegenerative Research\data\parkinsons\parkinsons_dataset"
    pk_out = r"d:\Machine Learning\Multimodal Neurodegenerative Research\data\parkinsons_data.zip"
    
    if os.path.exists(pk_out):
        os.remove(pk_out)
        
    zip_directory(pk_source, pk_out)
