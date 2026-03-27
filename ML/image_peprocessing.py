import cv2
import numpy as np
import os

def strong_preprocessing_pipeline(base_path):
    """
    Strong Preprocessing Pipeline for AI Wound Scanner
    Goal: High Accuracy (87%+) via standardized image quality.
    """
    categories = ['inflammation', 'proliferation', 'maturation']
    processed_data = []
    labels = []

    # Target size 256x256 as per project documentation [cite: 562]
    IMG_SIZE = 256 

    print("Starting Strong Preprocessing...")

    for label_idx, category in enumerate(categories):
        folder_path = os.path.join(base_path, category)
        
        for img_name in os.listdir(folder_path):
            try:
                img_path = os.path.join(folder_path, img_name)
                
                # 1. Image Acquisition [cite: 441]
                raw_img = cv2.imread(img_path)
                if raw_img is None: continue

                # 2. Resize to 256x256 [cite: 562, 568]
                # Standardizes input for the Deep Learning matrix
                resized_img = cv2.resize(raw_img, (IMG_SIZE, IMG_SIZE))

                # 3. Noise Reduction (Gaussian Blur) [cite: 210]
                # Removes 'sensor grain' from smartphone cameras [cite: 581]
                # (5,5) kernel is effective for clinical detail preservation
                denoised_img = cv2.GaussianBlur(resized_img, (5, 5), 0)

                # 4. Color Space Normalization [cite: 579]
                # pixel = pixel / 255.0 (Converts 0-255 to 0.0-1.0)
                # This makes training stable and prevents gradient issues
                normalized_img = denoised_img.astype('float32') / 255.0

                processed_data.append(normalized_img)
                labels.append(label_idx)

            except Exception as e:
                print(f"Error processing {img_name}: {e}")

    # Convert to NumPy arrays for Scikit-Learn/TensorFlow compatibility
    X = np.array(processed_data)
    y = np.array(labels)
    
    print(f"Preprocessing Complete. Total Images: {len(X)}")
    return X, y

# Execute using your local file path
DATASET_PATH = r"C:\AI WoundScanner Project\dataset"
X_train, y_train = strong_preprocessing_pipeline(DATASET_PATH)