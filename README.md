# Semantic Segmentation and Image Captioning (25/2/2022)
__Semantic segmentation__ attempts to partition the picture into semantically significant parts and to group each part into different classes. This varies from __instance
segmentation__ that is a parcel of a picture into a few ”coherent” parts yet with pracatically no effort to get what these parts address. Semantic segmentation advancements from this research apply to fields like autonomous vehicles, medical imaging, and satellite data analysis.

## Objective
The project explores advanced methods of semantic segmentation, selecting and implementing a Fully Convolutional Network (FCN) to train and test image datasets for accurate pixel-wise classification.

## Key Achievements 
- __Research__: Reviewed 40 papers on semantic segmentation methods, selecting FCN for its accuracy in maintaining image details.
- __Dataset__: Utilized the Cambridge CamVid dataset comprising 700 street-level images annotated with 11 semantic classes (e.g., roads, cars, pedestrians).
- __Methodology__: Designed a workflow including VGG-16 architecture for feature extraction and extensive preprocessing like histogram equalization, resizing, and class balancing.
- __Tools__: MATLAB with Deep Learning, Computer Vision, and Image Processing toolboxes.

## Steps in the Research Methodology
### 1. Setting up the VGG-16 Network:
- __Purpose:__ VGG-16, a pre-trained model, was selected for its superior feature extraction capabilities.
- __Features:__ VGG-16 processes input images of size 224x224x3 (RGB channels) and uses a sequence of small convolutional filters (3x3) and max pooling (2x2).
- __Advantages:__ Fewer parameters make it computationally efficient while maintaining accuracy.
- __Challenges:__ Training is resource-intensive, requiring significant computational power.
### 2. Labeling Data:
- The CamVid dataset was used, which already includes pixel-wise labeled images.
- __Process:__ MATLAB’s *Image Labeler* tool enables manual labeling, where each pixel is assigned a corresponding class.
- __Classes:__ The dataset's original 32 classes were reduced to 11 using grouping logic (e.g., all vehicle types classified under "Car").
### 3. Creating Datastores:
- __Purpose:__ Efficient management of large datasets.
- MATLAB’s *imageDatastore* and *pixelLabelDatastore* functions were employed.
- __Functionality:__ Original images and their labeled counterparts were stored in separate datastores for streamlined access during training and testing.
### 4. Loading Pixel-Labeled Images:
- Pixel-wise labeled data were prepared using the *pixelLabelDatastore* function.
- Labels were grouped into 11 categories, and corresponding RGB values were assigned to simplify semantic segmentation tasks.
### 5. Resizing the Dataset:
- All images were resized to a fixed resolution to ensure uniformity and avoid cropping issues during training.
- __Reason:__ FCN models require consistent input sizes to perform optimally.
### 6. Preparing Training and Testing Sets:
- The dataset was split:
  - __Training set:__ 70%
  - __Validation set:__ 15%
  - **Testing set:** 15%
### 7. Creating the Network:
- A Fully Convolutional Network (FCN) architecture was created using VGG-16 as the backbone.
- The model performs pixel-wise classification through encoder-decoder architecture:
   - **Encoder:** Extracts features from the input image.
   - **Decoder:** Upsamples the extracted features to match the original image size.
### 8. Balancing Classes Using Class Weights:
- **Problem:** Imbalance in class representation (e.g., more pixels labeled "Road" than "Pedestrian").
- **Solution:** Adjusted class weights during training to prevent bias towards over-represented classes.
### 9. Data Augmentation:
- Augmentation techniques like rotation, flipping, and cropping were applied to enhance dataset diversity and robustness of the model.
### 10. Setting Training Options:
- Configured hyperparameters such as learning rate, batch size, and optimization algorithm.
- Chose metrics like Intersection over Union (IoU) and loss functions (Binary Cross-Entropy and Focal Loss) for evaluation.
### 11. Training the Network:
- FCN-8s, known for its fine-grained upsampling capabilities, was employed.
- Used MATLAB’s deep learning framework for efficient training.
### 12. Testing and Evaluation:
- The trained network was tested using the test dataset.
- Performance metrics included:
  - **IoU:** Measures overlap between predicted and actual segmentation.
  - **Accuracy:** Percentage of correctly classified pixels.
## Results
- **A) IoU Scores:**
  - The IoU score evaluates the overlap between the predicted segmentation and the ground truth.
  - IoU scores were calculated for each class and averaged across all classes to measure segmentation performance.
    - **Class-wise IoU:** Results show high overlap for dominant classes like "Road" and "Car," and slightly lower for smaller or under-represented classes like "Pedestrian."
    - **Average IoU:** Around 92.8% ± 1.9% for most scenarios, depending on class distribution and model setup.
- **B) Accuracy:**
  - Accuracy reflects the percentage of pixels correctly classified into their respective classes.
  - Results indicated:
    - Overall pixel classification accuracy: **93.7%**.
    - Improved accuracy achieved through balancing class weights and data augmentation.
  - Specific improvements were noted for complex scenarios like differentiating between overlapping or noisy classes.
## Inferences
- **High performance:** The FCN-8s model demonstrated excellent segmentation capability, particularly in classes with consistent and sufficient representation.
- **Challenges:** Minor reductions in performance were observed for under-represented or ambiguous classes, mitigated by techniques like data augmentation and balanced class weights.
