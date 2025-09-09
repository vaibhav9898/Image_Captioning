# Image Captioning for Remote Sensing Images

A comprehensive implementation of CNN-LSTM based image captioning system specifically designed for remote sensing imagery using the RSICD dataset. This project demonstrates end-to-end pipeline for generating natural language descriptions of satellite and aerial images.

## 🚀 Project Overview

This project implements an image captioning system that automatically generates descriptive captions for remote sensing images. It uses a CNN-LSTM architecture where a CNN encoder extracts visual features from images and an LSTM decoder generates coherent captions describing the scene content including land use, structures, and spatial layouts.

## 📊 Dataset

**RSICD (Remote Sensing Image Captioning Dataset)**
- **Source**: Kaggle
- **Size**: 10,921 images with 5 captions each (54,605 total captions)
- **Splits**: 
  - Training: 8,736 images (43,680 captions)
  - Validation: 1,092 images (5,460 captions) 
  - Test: 1,093 images
- **Image Format**: RGB, resized to 224×224 pixels
- **Caption Stats**: Average 10.7 words per caption, vocabulary of 3,545 unique words

## 🏗️ Model Architecture

### CNN Encoder
- **Backbone**: ResNet-18 pre-trained on ImageNet
- **Feature Extraction**: 512-dimensional feature vectors
- **Fine-tuning**: Optional (configurable)

### LSTM Decoder
- **Architecture**: Multi-layer LSTM with attention mechanism
- **Embedding**: Word embeddings for vocabulary
- **Output**: Sequential word generation with beam search
- **Special Tokens**: `<START>`, `<END>`, `<PAD>`, `<UNK>`

### Key Features
- **Attention Mechanism**: Visual attention over spatial features
- **Feature Injection**: Multiple modes (concat, add, etc.)
- **Beam Search**: For improved caption generation
- **Teacher Forcing**: During training for faster convergence

## 📈 Performance Results

### Evaluation Metrics (BLEU Scores)
- **BLEU-1**: 0.5847
- **BLEU-2**: 0.4103  
- **BLEU-3**: 0.2968
- **BLEU-4**: 0.2238

### Model Specifications
- **Parameters**: ~2.1M trainable parameters
- **Training Time**: 50 epochs
- **Best Model**: Saved based on validation BLEU-4 score
- **Vocabulary Coverage**: 85.6% validation coverage, 14.4% OOV rate

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.8+
PyTorch 1.9+
torchvision
PIL (Pillow)
matplotlib
pandas
numpy
nltk
```

### Installation
```bash
# Clone the repository
git clone https://github.com/vaibhav9898/Image_Captioning.git
cd Image_Captioning

# Install dependencies
pip install torch torchvision pillow matplotlib pandas numpy nltk

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

### Dataset Setup
1. Download RSICD dataset from Kaggle
2. Extract to desired directory
3. Update `DATA_DIR` path in the notebook
4. Ensure directory structure:
   ```
   rsicd_export/
   ├── images/
   ├── captions_train.txt
   ├── captions_val.txt
   └── captions_test.txt
   ```

## 📝 Usage

### Training the Model
```python
# Configure hyperparameters
config = {
    'arch': 'resnet18',
    'feature_dim': 512,
    'embed_dim': 256,
    'hidden_dim': 512,
    'num_layers': 2,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 50
}

# Train the model
python train.py --config config.json
```

### Generating Captions
```python
# Load trained model
model = load_model('best_model.pth')

# Generate caption for an image
caption = generate_caption(model, image_path, vocab, max_length=20)
print(f"Generated Caption: {caption}")
```

### Evaluation
```python
# Evaluate on test set
bleu_scores = evaluate_model(model, test_loader, vocab)
print(f"BLEU-4 Score: {bleu_scores['bleu4']:.4f}")
```

## 🔍 Analysis & Explainability

### Grad-CAM Visualization
- **Purpose**: Visualize which image regions the model focuses on
- **Implementation**: Gradient-based attention maps overlaid on original images
- **Target**: EOS token logits for complete caption understanding

### Attention Analysis
- **Spatial Attention**: Shows where the model looks while generating each word
- **Temporal Analysis**: Tracks attention evolution across caption generation
- **Word-Region Mapping**: Links specific words to image regions

### Error Analysis
- **Common Failures**: Generic descriptions, object size sensitivity, vocabulary gaps
- **Case Studies**: Detailed analysis of mis-captioned examples
- **Improvement Suggestions**: Data augmentation, vocabulary expansion, multi-scale features

## 📁 Project Structure

```
Image_Captioning/
├── Image_Captioning.ipynb    # Main implementation notebook
├── README.md                 # Project documentation
└── .git/                     # Git repository
```

### Notebook Sections
1. **Data Analysis**: RSICD dataset exploration and statistics
2. **Preprocessing**: Image and text preprocessing pipelines
3. **Model Implementation**: CNN-LSTM architecture definition
4. **Training**: Model training with validation monitoring
5. **Evaluation**: BLEU score computation and analysis
6. **Explainability**: Grad-CAM and attention visualization
7. **Error Analysis**: Failure case studies and improvements

## 🎯 Key Features

### Data Processing
- ✅ Comprehensive dataset analysis and statistics
- ✅ Robust text preprocessing with vocabulary building
- ✅ Image augmentation and normalization
- ✅ Train/validation/test splits with no overlap

### Model Capabilities
- ✅ CNN-LSTM architecture with attention
- ✅ Multiple feature injection modes
- ✅ Beam search for caption generation
- ✅ Configurable hyperparameters

### Evaluation & Analysis
- ✅ Multiple BLEU score metrics
- ✅ Grad-CAM explainability
- ✅ Attention visualization
- ✅ Token importance analysis
- ✅ Error case studies

### Visualization
- ✅ Training/validation loss curves
- ✅ BLEU score progression
- ✅ Caption length histograms
- ✅ Vocabulary coverage analysis
- ✅ Attention heatmaps

## 🔬 Research Contributions

1. **Remote Sensing Focus**: Specialized implementation for satellite/aerial imagery
2. **Comprehensive Analysis**: End-to-end pipeline with detailed dataset analysis
3. **Explainability**: Multiple interpretation methods (Grad-CAM, attention, occlusion)
4. **Error Analysis**: Systematic study of failure cases and improvement strategies
5. **Reproducible Research**: Complete code with detailed documentation

## 📊 Results Summary

| Metric | Score |
|--------|-------|
| BLEU-1 | 0.5847 |
| BLEU-2 | 0.4103 |
| BLEU-3 | 0.2968 |
| BLEU-4 | 0.2238 |
| Vocab Size | 3,545 |
| Avg Caption Length | 10.7 words |
| Training Images | 8,736 |
| Model Parameters | ~2.1M |

## 🚧 Future Improvements

- [ ] **Transformer Architecture**: Replace LSTM with self-attention mechanisms
- [ ] **Multi-Scale Features**: Incorporate features from multiple CNN layers
- [ ] **Data Augmentation**: Advanced augmentation techniques for remote sensing
- [ ] **Vocabulary Expansion**: Include more descriptive terms and technical vocabulary
- [ ] **Cross-Domain Transfer**: Adapt model for different types of remote sensing imagery
- [ ] **Real-time Inference**: Optimize model for deployment and real-time applications

## 📚 References

1. RSICD Dataset - Remote Sensing Image Captioning Dataset
2. Lu et al. - "Exploring Models and Data for Remote Sensing Image Caption Generation"
3. Xu et al. - "Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"
4. He et al. - "Deep Residual Learning for Image Recognition"

## 📄 License

This project is developed for educational purposes as part of an Advanced Machine Learning course assignment.

## 🤝 Contributing

This is an academic project. For suggestions or discussions about the implementation, please feel free to open an issue or reach out to the author.

---

*This implementation serves as a comprehensive example of image captioning for remote sensing imagery, demonstrating both technical implementation and thorough analysis of the approach.*
