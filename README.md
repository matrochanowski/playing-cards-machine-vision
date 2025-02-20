# Playing Card Recognition with Constructive Neural Networks  
**Authors**: Mateusz Trochanowski (272478), Miłosz Rojek (272453)  

## Problem Description  
The goal of this project was to develop a model capable of recognizing playing cards in images. A key challenge was accurately identifying partially obscured cards or cards viewed from varying angles, reflecting how cards are typically held during social games.  

## Dataset Preparation  
The dataset consists of **22,000 images**, each annotated with:  
- Class labels  
- Normalized bounding box coordinates in the format `(label, x_center, y_center, width, height)`  
Each image contains **3 to 5 cards**.  

## Model Training  
### Binary Model (Card Presence Detection)  
A binary classifier detects whether a card exists in an image region.  
- **Input**: `128x128x3` RGB patches  
- **Preprocessing**: Undersampling to balance classes, normalization  
- **Architecture**: Custom CNN with convolutional layers and fully connected layers  
- **Accuracy**: Achieved satisfactory performance in distinguishing card/non-card regions  

### Classification Model (Card Label Prediction)  
A CNN classifies detected cards into 52 possible classes (suits and ranks).  
- **Architecture**:  
  - 5 convolutional blocks (32 → 512 channels) with `2x2` pooling  
  - Fully connected layers (512 → 256 → 52 neurons)  
  - Dropout for regularization  
- **Output**: Probability distribution over card classes  

## Constructive Localization Approach  
A multi-stage process to detect cards without traditional object detection frameworks:  
1. **Image Splitting**: Divide input image into candidate regions  
2. **Multi-Level Analysis**:  
   - Use binary model to identify candidate regions  
   - Recursively split positive regions for finer localization  
3. **Bounding Box Generation**:  
   - Merge overlapping detections  
   - Scale coordinates to original image dimensions  
4. **Result Aggregation**: Sort detections by confidence  

## Results & Performance  
**F1-Score**: `0.55`  
**Key Observations**:  
- Correctly detects all visible cards in **74%** of test cases  
- Struggles with duplicate detections and similar-looking classes (e.g., "KS" vs "KC")  

### Example Detections  
![Detected Cards 1](karty1.png)  
`Predicted: ['8D', '10C', '2S']`  

![Detected Cards 2](karty2.png)  
`Predicted: ['KS', 'JH', '9C']`  

![Detected Cards 3](karty3.png)  
`Predicted: ['9C', 'KS', '4C', 'KC', 'AD']`  

## Metrics  
**F1-Score**: Harmonic mean of precision and recall:  
