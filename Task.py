import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_squared_error

categories = ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']
path1 = "C:/Users/dell/Downloads/wbc_data/Train/"
path2= "C:/Users/dell/Downloads/wbc_data/Test/"

hog_values = {
    'orientations':      9,
    'pixels_per_cell':  (8, 8),
    'cells_per_block':  (2, 2),
    'block_norm':      'L2-Hys',
    'transform_sqrt':   True,
    'feature_vector':   True
}

def center_crop(image):
    row, col, c = image.shape
    startx = row // 2 - 300 // 2
    starty = col // 2 - 300 // 2
    return image[starty : starty + 300, startx : startx + 300]

# Preprocessing
def apply_preprocess_hog(image):
    out_im = center_crop(image)
    out_im1 = cv2.cvtColor(out_im, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(out_im1, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(out_im1, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(sobelx, sobely)
    mag = cv2.convertScaleAbs(mag)
    feature_vector = hog(mag, **hog_values)
    return feature_vector

# Computing average for all feature vectors
def average_hog():
    avg_feature_vector = {}
    lengths = {}
    for cat in categories:
        total_features = []
        for i in range(1, 100):
            final_path = path1 + cat + "/" + cat + "_" + str(i) + ".jpg"
            input_image  = cv2.imread(final_path)
            features = apply_preprocess_hog(input_image)
            total_features.append(features)
        max_len = max(len(f) for f in total_features)
        padded = [np.pad(f, (0, max_len - len(f))) for f in total_features]
        avg = np.mean(padded, axis=0)
        avg_feature_vector[cat] = avg
        lengths[cat] = len(avg)
    print("Average feature‑vector lengths per class: ", lengths)
    return avg_feature_vector

# Testing images
def test_image(image, avg_features):
    extracted_features = apply_preprocess_hog(image)
    max_len = max(len(v) for v in avg_features.values())
    feature_vector = np.pad(extracted_features, (0, max_len - len(extracted_features)))
    errors = {}
    for cat in categories:
        out = avg_features[cat]
        error = mean_squared_error(feature_vector, out)
        errors[cat] = error
    best_class = None
    best_error = None
    for cat, error in errors.items():
        if best_error is None or error < best_error:
            best_error = error
            best_class = cat
    return best_class

# Testing the dataset
def test_set(avg_features):
    true_classes      = []
    predicted_classes = []
    for cat in categories:
        for i in range(1, 21):
            path = path2 + cat + "/" + cat + "_" + str(i) + ".jpg"
            input_image = cv2.imread(path)
            prediction = test_image(input_image, avg_features)
            true_classes.append(cat)
            predicted_classes.append(prediction)

    out_im = confusion_matrix(true_classes, predicted_classes, labels=categories)
    return out_im

def display_hog(avg_features, length=255):
    for cat, vec in avg_features.items():
        plt.figure(figsize=(6,3))
        plt.plot(vec[:length])
        plt.title(cat + " - Avg HoG (first " + str(length) + " values)")
        plt.xlabel("Feature Index")
        plt.ylabel("Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_confusion_matrix(mat):
    disp = ConfusionMatrixDisplay(confusion_matrix=mat, display_labels=categories)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

avg_templates = average_hog()
display_hog(avg_templates)
out_matrix = test_set(avg_templates)
plot_confusion_matrix(out_matrix)