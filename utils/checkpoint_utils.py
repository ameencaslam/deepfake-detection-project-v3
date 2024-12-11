import os
import shutil
from datetime import datetime

def zip_checkpoints():
    """Create a zip archive of checkpoints in Kaggle environment"""
    # Kaggle paths
    source_dir = "/kaggle/working/checkpoints"
    output_dir = "/kaggle/working"
    
    # Generate zip name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"checkpoints_{timestamp}"
    zip_path = os.path.join(output_dir, f"{zip_name}.zip")
    
    try:
        # Create zip archive
        if os.path.exists(source_dir):
            shutil.make_archive(
                os.path.splitext(zip_path)[0],
                'zip',
                source_dir
            )
            print(f"\nCheckpoints successfully zipped to: {zip_path}")
            return zip_path
        else:
            print(f"\nNo checkpoints found in {source_dir}")
            return None
    except Exception as e:
        print(f"\nError creating zip archive: {str(e)}")
        return None

def extract_checkpoints(zip_path):
    """Extract a checkpoint zip archive in Kaggle environment"""
    extract_dir = "/kaggle/working/checkpoints"
    
    try:
        # Create extract directory if it doesn't exist
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extract the archive
        shutil.unpack_archive(zip_path, extract_dir)
        print(f"\nCheckpoints extracted to: {extract_dir}")
        return extract_dir
    except Exception as e:
        print(f"\nError extracting zip archive: {str(e)}")
        return None

if __name__ == "__main__":
    # When run as a script, just zip the checkpoints
    zip_checkpoints() 