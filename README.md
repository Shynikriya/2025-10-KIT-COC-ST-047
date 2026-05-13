# 2025-10-KIT-COC-ST-047
Intelligent Identification of Traditional Village Landscape Genes Using Deep Learning for Landscape Architecture and Human Settlements


Algorithm 1: Village Landscape Gene Identification
Input: Aerial image dataset (LandCover.ai)
Output: Predicted landscape genes with heatmaps
1. Load dataset and annotations
2. For each image in dataset:
       a. Resize to 280x224 pixels
       b. Normalize pixel values
       c. One-hot encode labels
       d. Apply data augmentation (flip, rotate, HSV adjust)
3. Initialize VGG16 pretrained on ImageNet
4. Freeze first 5 convolution layers
5. Fine-tune remaining layers with dataset
6. Train model using SGD optimizer for ~120 epochs
7. For each test image:
       a. Predict landscape gene classes
       b. Generate heatmap for spatial visualization
8. Evaluate model using accuracy, precision, recall, F1-score
9. Output predictions and visualizations


