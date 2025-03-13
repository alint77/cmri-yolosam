import os
import sys
import nibabel as nib
import numpy as np
import yaml
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def create_symlinks(source_dir, target_dir):
    """Creates symbolic links for the training and testing directories."""
    os.makedirs(target_dir, exist_ok=True)
    for subdir in ['training', 'testing']:
        source_path = os.path.join(source_dir, subdir)
        target_path = os.path.join(target_dir, subdir)
        if not os.path.exists(target_path):
            if os.path.exists(source_path):  # Check if source exists before linking
                os.symlink(source_path, target_path)
                print(f"Created symlink: {target_path} -> {source_path}")
            else:
                print(f"Warning: Source directory not found: {source_path}", file=sys.stderr)
        else:
            print(f"Symlink already exists: {target_path}")

def load_nifti_data(filepath):
    """Loads NIfTI data and returns the image array."""
    img = nib.load(filepath)
    return img.get_fdata()

def get_bounding_box(mask):
    """Calculates the bounding box coordinates from a segmentation mask."""
    coords = np.where(mask != 0)
    if len(coords[0]) == 0:  # Handle empty masks
        return None  
    x_min, y_min = np.min(coords[1]), np.min(coords[0])
    x_max, y_max = np.max(coords[1]), np.max(coords[0])
    return x_min, y_min, x_max, y_max

def convert_to_yolo_format(bbox, image_width, image_height):
    """Converts bounding box coordinates to YOLO format."""
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / (2 * image_width)
    y_center = (y_min + y_max) / (2 * image_height)
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return x_center, y_center, width, height

def save_image_slice(image_slice, output_path):
    """Saves a 2D image slice as a PNG file."""
    plt.imsave(output_path, image_slice, cmap='gray')

def process_patient_data(patient_dir, output_dir, split):
    """Processes data for a single patient."""
    patient_id = os.path.basename(patient_dir)
    output_dir = os.path.join(output_dir, split, patient_id)
    os.makedirs(output_dir, exist_ok=True)

    info_file = os.path.join(patient_dir, 'Info.cfg')
    config = {}
    try:
        with open(info_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line: # ignore empty lines
                    key, value = line.split(': ')
                    config[key] = value
    except FileNotFoundError:
        print(f"Warning: Info.cfg not found for patient {patient_id}. Skipping.")
        return
    
    try:
        ed_frame = int(config['ED'])
        es_frame = int(config['ES'])
    except KeyError:
        print(f"Warning: ED or ES frame not found in {info_file}. Skipping patient.")
        return
    
    for frame_num in [ed_frame, es_frame]:
        try:
            img_path = os.path.join(patient_dir, f"{patient_id}_frame{frame_num:02d}.nii.gz")
            mask_path = os.path.join(patient_dir, f"{patient_id}_frame{frame_num:02d}_gt.nii.gz")

            if not os.path.exists(img_path) or not os.path.exists(mask_path):
                print(f"Warning: Missing image or mask file for {patient_id}, frame {frame_num}. Skipping.")
                continue

            img_data = load_nifti_data(img_path)
            mask_data = load_nifti_data(mask_path)

            # Handle potential 3D masks (select the first slice if so)
            if len(mask_data.shape) == 3:
                num_slices = mask_data.shape[2]
            else:
                print("mask data shape", mask_data.shape)
                raise ValueError("Unexpected mask data dimensions.")

            for slice_idx in range(num_slices):
                img_slice = img_data[:, :, slice_idx]  # Assuming 2D images
                mask_slice = mask_data[:, :, slice_idx]
                image_height, image_width = img_slice.shape

                label_file = os.path.join(output_dir, f"{patient_id}_frame{frame_num:02d}_slice{slice_idx:02d}.txt")
                image_file = os.path.join(output_dir, f"{patient_id}_frame{frame_num:02d}_slice{slice_idx:02d}.png")
                save_image_slice(img_slice, image_file)

                with open(label_file, 'w') as f:
                    for label in [1, 2, 3]:  # RV, myocardium, LV
                        bbox = get_bounding_box(mask_slice == label)
                        if bbox:
                            yolo_bbox = convert_to_yolo_format(bbox, image_width, image_height)
                            f.write(f"{label-1} {' '.join(map(str, yolo_bbox))}\n")  # YOLO class starts from 0

        except FileNotFoundError:
            print(f"Warning: Could not process patient {patient_id}, frame {frame_num}. Skipping.")
            continue
        except ValueError as e:
            print(f"Error processing patient {patient_id}, frame{frame_num}: {e}")
            continue

def create_dataset_yaml(data_dir, train_dir, val_dir, test_dir):
    """Creates the dataset YAML file."""
    data = {
        'path': data_dir,
        'train': train_dir,
        'val': val_dir,
        'test': test_dir,
        'nc': 3,
        'names': ['RV', 'myocardium', 'LV']
    }
    yaml_path = os.path.join(data_dir, 'cardiac_mri.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    print(f"Created dataset YAML file: {yaml_path}")


# Main script execution
if __name__ == "__main__":
    root_dir = os.getcwd()  # Get the current working directory
    source_data_dir = os.path.join(root_dir, 'database')
    target_data_dir = os.path.join(root_dir, 'sam2_project', 'data')
    yolo_data_dir = os.path.join(target_data_dir, 'yolo_data')

    create_symlinks(source_data_dir, target_data_dir)

    # Create training and validation splits
    training_patients = [p for p in os.listdir(os.path.join(target_data_dir, 'training')) if os.path.isdir(os.path.join(target_data_dir, 'training', p))]
    train_patients, val_patients = train_test_split(training_patients, test_size=0.2, random_state=42) # 80/20 split

    # Process data for training, validation, and testing sets
    for patient in train_patients:
        process_patient_data(os.path.join(target_data_dir, 'training', patient), yolo_data_dir, 'train')
    for patient in val_patients:
        process_patient_data(os.path.join(target_data_dir, 'training', patient), yolo_data_dir, 'val')
    for patient in os.listdir(os.path.join(target_data_dir, 'testing')):
        if os.path.isdir(os.path.join(target_data_dir, 'testing', patient)):
            process_patient_data(os.path.join(target_data_dir, 'testing', patient), yolo_data_dir, 'test')
            

    # Create dataset YAML file
    create_dataset_yaml(
        target_data_dir,
        'yolo_data/train',
        'yolo_data/val',
        'yolo_data/test'
    )
    print("Data preparation script completed.")