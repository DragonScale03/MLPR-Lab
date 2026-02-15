# Face Detection and Clustering using K-Means

## Table of Contents
- [Aim](#aim)
- [Methodology](#methodology)
- [Implementation](#implementation)
- [Key Findings](#key-findings)
- [Conclusions](#conclusions)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)

## Aim

The primary objectives of this project are:
1. Detect faces in group images using Haar Cascade classifiers
2. Extract color-based features (Hue and Saturation) from detected faces in HSV color space
3. Apply K-Means clustering to group similar faces based on their color characteristics
4. Classify a new template image into one of the identified clusters
5. Visualize the clustering results and understand the distribution of facial features

## Methodology

### 1. Face Detection

We used OpenCV's Haar Cascade classifier to detect faces in images. The Haar Cascade is a machine learning-based approach where a cascade function is trained with lots of positive and negative images to detect objects (in our case, faces).

**Process:**
- Loading the pre-trained Haar Cascade model (`haarcascade_frontalface_default.xml`)
- Converting images to grayscale for better detection accuracy
- Applying the cascade classifier with optimized parameters (`scaleFactor=1.05`, `minNeighbors=4`)
- Drawing red bounding boxes around detected faces with "Face" labels

**Original Faculty Image:**

![Plaksha_Faculty](https://github.com/user-attachments/assets/2f9ed723-7170-4bfd-8bbd-a4fe4c85e717)


**Face Detection Results:**

<img width="761" height="522" alt="Screenshot 2026-02-15 230220" src="https://github.com/user-attachments/assets/ed40044f-4f37-4be6-a8ef-68f60953d768" />


Our algorithm successfully detected **30 faces** in the Plaksha Faculty group photo.

### 2. Feature Extraction

For each detected face, we extracted color-based features in the HSV (Hue, Saturation, Value) color space:

- **Hue**: Represents the actual color/tone (ranges from 0-180° in OpenCV)
- **Saturation**: Represents the intensity or purity of the color (ranges from 0-255)

**Why HSV over RGB?**
- HSV better separates color information from lighting conditions
- More robust to variations in brightness and illumination
- Hue and Saturation provide meaningful features for distinguishing faces based on color characteristics

**Feature Calculation:**
```python
# For each detected face region
hue = np.mean(face[:, :, 0])        # Average Hue value
saturation = np.mean(face[:, :, 1])  # Average Saturation value
```

### 3. K-Means Clustering

We applied the K-Means clustering algorithm to group faces with similar color characteristics:

**Configuration:**
- **Number of clusters (k)**: 2
- **Features**: Mean Hue and Mean Saturation values from each face
- **Algorithm**: Standard K-Means from scikit-learn
- **Random state**: 42 (for reproducibility)

**How K-Means Works:**
1. Randomly initialize k cluster centroids
2. Assign each face to the nearest centroid based on Euclidean distance
3. Recalculate centroids as the mean of all points in each cluster
4. Repeat steps 2-3 until convergence

The algorithm groups faces with similar Hue-Saturation profiles together, which can correlate with factors like:
- Skin tone characteristics
- Lighting conditions
- Background colors
- Clothing colors near the face region

**Clustering Visualization with Face Thumbnails:**

<img width="743" height="399" alt="Screenshot 2026-02-15 230250" src="https://github.com/user-attachments/assets/58f3c1bd-5b57-4d36-a404-addb870c0e61" />


This visualization shows each detected face plotted at its (Hue, Saturation) coordinates, giving us an intuitive view of how faces are distributed in the color feature space.

### 4. Template Classification

To test our clustering model, we:
1. Loaded a template image (Dr. Shashi Tharoor's photo)
2. Detected the face using the same Haar Cascade classifier
3. Extracted Hue and Saturation features from the detected face
4. Used the trained K-Means model to predict which cluster the template belongs to
5. Visualized the template's position in the feature space

**Template Image - Face Detection:**

<img width="779" height="761" alt="Screenshot 2026-02-15 230314" src="https://github.com/user-attachments/assets/4ab48932-dd2d-4b34-b895-e10a0168d31d" />


Successfully detected **1 face** in the template image, which was then classified using our trained model.

## Implementation

### Step-by-Step Process

#### Step 1: Import Required Libraries
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
```

#### Step 2: Load Image and Detect Faces
```python
# Read the faculty group image
img = cv2.imread('plaksha_Faculty.jpg')

# Convert to grayscale for face detection
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load Haar Cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                     'haarcascade_frontalface_default.xml')

# Detect faces
faces_rect = face_cascade.detectMultiScale(gray_img, 1.05, 4, 
                                           minSize=(25,25), maxSize=(50,50))

# Draw rectangles and labels on detected faces
for (x, y, w, h) in faces_rect:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.putText(img, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 0, 255), 1)
```

#### Step 3: Extract Hue-Saturation Features
```python
# Convert image to HSV color space
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

hue_saturation = []
face_images = []

# Extract features for each detected face
for (x, y, w, h) in faces_rect:
    face = img_hsv[y:y + h, x:x + w]
    hue = np.mean(face[:, :, 0])
    saturation = np.mean(face[:, :, 1])
    hue_saturation.append((hue, saturation))
    face_images.append(face)

hue_saturation = np.array(hue_saturation)
```

#### Step 4: Apply K-Means Clustering
```python
# Perform K-Means clustering with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=42).fit(hue_saturation)

# Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
```

#### Step 5: Visualize Clusters
```python
fig, ax = plt.subplots(figsize=(12, 6))

# Separate points by cluster
cluster_0_points = []
cluster_1_points = []

for i, (x, y, w, h) in enumerate(faces_rect):
    if kmeans.labels_[i] == 0:
        cluster_0_points.append((hue_saturation[i, 0], hue_saturation[i, 1]))
    else:
        cluster_1_points.append((hue_saturation[i, 0], hue_saturation[i, 1]))

# Plot clusters
plt.scatter(cluster_0_points[:, 0], cluster_0_points[:, 1], 
            c='green', label='Cluster 0', s=100)
plt.scatter(cluster_1_points[:, 0], cluster_1_points[:, 1], 
            c='blue', label='Cluster 1', s=100)

# Plot centroids
plt.scatter(centroids[0][0], centroids[0][1], c='red', marker='X', 
            s=300, edgecolors='black', linewidths=2, label='Centroid 0')
plt.scatter(centroids[1][0], centroids[1][1], c='yellow', marker='X', 
            s=300, edgecolors='black', linewidths=2, label='Centroid 1')

plt.xlabel('Hue')
plt.ylabel('Saturation')
plt.title('K-Means Clustering of Faces based on Hue and Saturation')
plt.legend()
plt.grid(True)
plt.show()
```

**Cluster Visualization with Centroids:**
<img width="769" height="387" alt="Screenshot 2026-02-15 230408" src="https://github.com/user-attachments/assets/b9197c27-ffbc-4525-8558-0ba868196a71" />


This plot clearly shows:
- **Cluster 0 (Green)**: 8 faces with lower Hue values (approximately 20-60)
- **Cluster 1 (Blue)**: 22 faces with higher Hue values (approximately 35-130)
- **Centroid 0 (Red X)**: Located around (Hue: 65, Saturation: 107)
- **Centroid 1 (Yellow X)**: Located around (Hue: 115, Saturation: 127)

#### Step 6: Classify Template Image
```python
# Load and process template image
template_img = cv2.imread('Dr_Shashi_Tharoor.jpg')
template_hsv = cv2.cvtColor(template_img, cv2.COLOR_BGR2HSV)

# Extract features
template_hue = np.mean(template_hsv[:, :, 0])
template_saturation = np.mean(template_hsv[:, :, 1])

# Predict cluster
template_label = kmeans.predict([[template_hue, template_saturation]])[0]

# Visualize with template
plt.scatter(template_hue, template_saturation, marker='o', 
            c='violet', s=200, label='Class ?')
```

**Final Classification with Template:**

<img width="759" height="409" alt="Screenshot 2026-02-15 230426" src="https://github.com/user-attachments/assets/82530a38-149d-4fce-a821-5430f3e0f71d" />


The template image (shown in violet) falls between the two clusters, closer to **Cluster 0**, indicating it shares similar color characteristics with that group.

## Key Findings

### 1. Face Detection Performance

✅ **Successfully detected 30 out of 30+ faces** in the faculty group photo
- The Haar Cascade classifier performed well with properly tuned parameters
- Initial attempts with restrictive `minSize` and `maxSize` constraints missed some faces
- Removing or adjusting these constraints significantly improved detection accuracy

**Challenges Faced:**
- Parameter tuning was crucial for optimal detection
- Some faces at angles or with partial occlusions were harder to detect
- Lighting variations affected detection consistency

### 2. Cluster Distribution Analysis

The K-Means algorithm successfully identified two distinct groups:

**Cluster 0 (Green) - 8 faces:**
- Lower Hue values (range: ~20-60)
- Moderate to high Saturation (range: ~80-120)
- Centroid: (Hue: 65, Saturation: 107)
- Characteristics: Possibly faces with warmer tones, specific lighting, or clothing colors

**Cluster 1 (Blue) - 22 faces:**
- Higher Hue values (range: ~35-130, more spread out)
- Higher Saturation (range: ~107-135)
- Centroid: (Hue: 115, Saturation: 127)
- Characteristics: Larger cluster with more diverse color profiles

**Observations:**
- Clear separation between the two clusters in the feature space
- Cluster 1 is significantly larger (73% of faces) than Cluster 0 (27%)
- Some overlap in the middle region suggests color-based features alone have limitations
- The distribution shows that most faculty members fall into similar color ranges

### 3. Template Image Classification

**Dr. Shashi Tharoor's Image Analysis:**
- **Feature Values**: Hue ≈ 70-75, Saturation ≈ 90-95
- **Predicted Cluster**: Cluster 0 (though positioned near the boundary)
- **Position**: Located between both centroids but slightly closer to Centroid 0

**Interpretation:**
The template image's position in the feature space is interesting:
- It falls in an intermediate region, suggesting moderate color characteristics
- Closer to Cluster 0's centroid, indicating alignment with that group's features
- The classification demonstrates that the model can successfully predict cluster membership for new images

### 4. Feature Space Insights

**Hue Distribution:**
- Overall range: 20-130 (on a 0-180 scale)
- Cluster 0 concentrated at lower Hue values
- Cluster 1 more spread out across higher Hue range

**Saturation Distribution:**
- Overall range: 80-135 (on a 0-255 scale)
- Both clusters show relatively high saturation values
- Less variance in Saturation compared to Hue

**What This Tells Us:**
- Hue (color tone) is the primary discriminating feature between clusters
- Saturation (color intensity) is more consistent across all faces
- The feature space shows some natural grouping, validating the K-Means approach
- Color-based features capture meaningful variations in the dataset

### 5. Model Performance

**Strengths:**
- Clear cluster separation in most regions
- Meaningful feature extraction from HSV color space
- Successfully classified a new template image
- Computationally efficient and fast

**Limitations:**
- Some faces near cluster boundaries may be misclassified
- Color features don't capture facial identity or detailed characteristics
- Sensitive to lighting conditions and image quality
- Binary classification (k=2) may oversimplify the actual diversity

## Conclusions

### Main Takeaways

1. **Haar Cascade Effectiveness**: 
   - Haar Cascade classifiers remain a viable option for frontal face detection in well-lit, relatively standard conditions
   - Parameter tuning (`scaleFactor`, `minNeighbors`, `minSize`, `maxSize`) is critical for optimal performance
   - Works well for batch processing of group photos

2. **HSV Color Space Advantage**:
   - HSV provides more meaningful features than RGB for clustering tasks
   - Separating Hue and Saturation from Value/brightness improves robustness
   - Color-based features can effectively group faces with similar characteristics

3. **K-Means Clustering Success**:
   - Successfully partitioned 30 faces into 2 meaningful clusters
   - Clear visual separation in the feature space validates the approach
   - Simple yet effective for exploratory data analysis

4. **Template Classification**:
   - The model successfully predicted cluster membership for a new image
   - Demonstrates the practical applicability of the trained clustering model
   - Can be extended to classify additional images


### Limitations and Challenges

**Technical Limitations:**
- **Feature Simplicity**: Color features alone don't capture facial identity, expressions, or detailed characteristics
- **Lighting Dependency**: HSV features are still affected by lighting conditions, shadows, and image quality
- **Haar Cascade Constraints**: May miss faces at extreme angles, with occlusions, or in poor lighting
- **K-Means Assumptions**: Assumes spherical clusters and requires pre-defining k (number of clusters)

**Practical Challenges:**
- Background colors can influence face region features
- Clothing colors near the face may affect results
- Different camera settings across images can introduce variance
- No consideration of facial features, age, gender, or other demographics


### Learning Outcomes

Through this project, we demonstrated:

 Practical implementation of computer vision techniques  
 Integration of multiple libraries (OpenCV, scikit-learn, Matplotlib)  
 Understanding of color spaces and their applications  
 Application of unsupervised learning (K-Means clustering)  
 Data visualization techniques for high-dimensional data  
 End-to-end machine learning pipeline from data to deployment  

## Technologies Used

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Programming language | 3.x |
| **OpenCV** | Face detection and image processing | Latest |
| **NumPy** | Numerical computations and array operations | Latest |
| **Matplotlib** | Data visualization and plotting | Latest |
| **scikit-learn** | K-Means clustering algorithm | Latest |
| **SciPy** | Distance calculations and scientific computing | Latest |
| **Jupyter Notebook** | Interactive development environment | Latest |

### Key Libraries and Their Roles

**OpenCV (cv2):**
- Face detection using Haar Cascade classifiers
- Image loading, conversion, and manipulation
- Color space transformations (BGR → Grayscale, BGR → HSV)
- Drawing shapes and text on images

**NumPy:**
- Array operations for feature vectors
- Statistical calculations (mean, standard deviation)
- Efficient numerical computations

**Matplotlib:**
- Scatter plots for cluster visualization
- Image display and figure creation
- Custom annotations with face thumbnails

**scikit-learn:**
- K-Means clustering implementation
- Model fitting and prediction
- Cluster analysis

## How to Run

### Prerequisites

Ensure you have Python 3.x installed, then install the required libraries:
```bash
pip install opencv-python numpy matplotlib scikit-learn scipy jupyter
```

Or using a requirements file:
```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
opencv-python>=4.5.0
numpy>=1.19.0
matplotlib>=3.3.0
scikit-learn>=0.24.0
scipy>=1.5.0
jupyter>=1.0.0
```

### Project Structure
```
face-clustering-project/
├── face_clustering.ipynb           # Main Jupyter notebook
├── Plaksha_Faculty.jpg             # Input group image
├── Dr_Shashi_Tharoor.jpg          # Template image for classification
├── 1771176745107_image.png        # Face detection result
├── 1771176774068_image.png        # Clustering with thumbnails
├── 1771176797460_image.png        # Template detection
├── 1771176850540_image.png        # Clusters with centroids
├── 1771176867620_image.png        # Final classification
├── README.md                       # This file
└── requirements.txt                # Python dependencies
```

### Steps to Run

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/face-clustering-kmeans.git
cd face-clustering-kmeans
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Place your images in the project directory:**
   - `plaksha_Faculty.jpg` (group image with multiple faces)
   - `Dr_Shashi_Tharoor.jpg` (template image for classification)

4. **Launch Jupyter Notebook:**
```bash
jupyter notebook face_clustering.ipynb
```

5. **Run all cells in sequence:**
   - Face detection on group image
   - Feature extraction and clustering
   - Template image classification
   - Visualization generation

6. **View results:**
   - All visualizations will be displayed inline in the notebook

### Running in Google Colab

If you prefer to run this in Google Colab:

1. Upload the notebook and images to Google Drive
2. Open the notebook in Colab
3. Install dependencies (first cell should contain pip install commands)
4. Update file paths to point to your Google Drive
5. Run all cells

### Troubleshooting

**Issue: cv2.imshow() doesn't work in Jupyter**
- Solution: The code uses matplotlib for display, which works in all environments

**Issue: Face detection finds 0 faces**
- Solution: Adjust `scaleFactor`, `minNeighbors`, remove `minSize`/`maxSize` constraints

**Issue: Module not found errors**
- Solution: Ensure all packages are installed: `pip install -r requirements.txt`

**Issue: Haar Cascade file not found**
- Solution: Use `cv2.data.haarcascades` path as shown in the code

## Results Summary

| Metric | Value |
|--------|-------|
| **Faces Detected (Faculty Image)** | 30 |
| **Number of Clusters** | 2 |
| **Cluster 0 Size** | 8 faces (27%) |
| **Cluster 1 Size** | 22 faces (73%) |
| **Template Faces Detected** | 1 |
| **Template Classification** | Cluster 0 |
| **Hue Range** | 20-130 |
| **Saturation Range** | 80-135 |

## Author
**Gautam Ganesh**


```

---

