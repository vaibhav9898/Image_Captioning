# Image Captioning for Remote Sensing Images

A comprehensive implementation of image captioning systems specifically designed for remote sensing imagery using the RSICD dataset. This project demonstrates end-to-end pipelines for generating natural language descriptions of satellite and aerial images using both traditional CNN-LSTM and modern CNN-Transformer architectures.

## 🚀 Project Overview

This project implements multiple image captioning architectures that automatically generate descriptive captions for remote sensing images. The implementation includes both traditional CNN-LSTM and modern CNN-Transformer architectures where CNN encoders extract visual features from images and different decoder architectures generate coherent captions describing the scene content including land use, structures, and spatial layouts.

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

## 🏗️ Model Architectures

This project implements and compares two different decoder architectures:

### CNN Encoder
- **Backbone**: ResNet-18 pre-trained on ImageNet
- **Feature Extraction**: 512-dimensional feature vectors
- **Fine-tuning**: Optional (configurable)

### Decoder Options

#### 1. LSTM Decoder
- **Architecture**: Multi-layer LSTM 
- **Embedding**: Word embeddings for vocabulary
- **Feature Injection**: Multiple modes (init_hidden, img_token)
- **Output**: Sequential word generation with greedy/beam search
- **Special Tokens**: `<START>`, `<END>`, `<PAD>`, `<UNK>`

#### 2. Transformer Decoder
- **Architecture**: Multi-layer Transformer decoder with self-attention
- **Attention Mechanism**: Multi-head self-attention (8 heads)
- **Layers**: 2 transformer decoder layers
- **Embedding Dimension**: 384
- **Positional Encoding**: Learned positional embeddings
- **Memory Integration**: Image features projected to memory tokens
- **Output**: Sequential word generation with causal masking

### Key Features
- **Multiple Architectures**: Both LSTM and Transformer decoders
- **Attention Mechanism**: Visual attention over spatial features (Transformer)
- **Feature Injection**: Multiple modes (concat, add, etc.)
- **Beam Search**: For improved caption generation
- **Teacher Forcing**: During training for faster convergence

## 📈 Performance Results

### CNN + LSTM Decoder (Cached Features)
- **BLEU-1**: 0.5847
- **BLEU-2**: 0.4103  
- **BLEU-3**: 0.2968
- **BLEU-4**: 0.2238

### CNN + Transformer Decoder (End-to-End)
- **BLEU-4**: 0.2074
- **METEOR**: 0.3577
- **Caption Length**: 10.50 ± 2.88 words
- **Degenerate Repetitions**: 0.27%

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

### Training Models

#### For LSTM Decoder
```python
# Configure hyperparameters
config = {
    'arch': 'resnet18',
    'decoder': 'lstm',
    'feature_dim': 512,
    'embed_dim': 384,
    'hidden_dim': 512,
    'num_layers': 2,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 50,
    'lstm_injection_mode': 'init_hidden'
}
```

#### For Transformer Decoder
```python
# Configure hyperparameters
config = {
    'arch': 'resnet18',
    'decoder': 'transformer',
    'feature_dim': 512,
    'embed_dim': 384,
    'num_layers': 2,
    'nhead': 8,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'epochs': 50,
    'transformer_memory_tokens': 1
}
```

### Generating Captions
```python
# Load trained model (LSTM example)
model = load_model('models/resnet18_lstm/best_model.pth')

# Load trained model (Transformer example)  
model = load_model('models/resnet18_transformer/best_model.pth')

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
3. **Model Implementation**: CNN encoder with LSTM/Transformer decoder options
4. **Training**: Model training with validation monitoring for both architectures
5. **Evaluation**: BLEU score computation and analysis for both models
6. **Explainability**: Grad-CAM and attention visualization
7. **Error Analysis**: Failure case studies and improvements

## 🎯 Key Features

### Data Processing
- ✅ Comprehensive dataset analysis and statistics
- ✅ Robust text preprocessing with vocabulary building
- ✅ Image augmentation and normalization
- ✅ Train/validation/test splits with no overlap

### Model Capabilities
- ✅ CNN-LSTM architecture with multiple injection modes
- ✅ CNN-Transformer architecture with self-attention
- ✅ Multi-head attention mechanism (Transformer)
- ✅ Positional encoding for sequence modeling
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
3. **Multiple Architectures**: Comparison between LSTM and Transformer decoders
4. **Explainability**: Multiple interpretation methods (Grad-CAM, attention, occlusion)
5. **Error Analysis**: Systematic study of failure cases and improvement strategies
6. **Reproducible Research**: Complete code with detailed documentation

## 📊 Results Summary

### Model Comparison

| Architecture | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | Parameters |
|--------------|--------|--------|--------|--------|--------|------------|
| CNN + LSTM (Cached) | 0.5847 | 0.4103 | 0.2968 | 0.2238 | - | ~2.1M |
| CNN + Transformer | - | - | - | 0.2074 | 0.3577 | ~2.1M |

### Dataset Statistics
- **Vocab Size**: 3,545
- **Avg Caption Length**: 10.7 words  
- **Training Images**: 8,736
- **Validation Coverage**: 85.6%

## 🚧 Future Improvements

- [ ] **Advanced Transformer Variants**: Implement Vision Transformer (ViT) encoder
- [ ] **Multi-Scale Features**: Incorporate features from multiple CNN layers
- [ ] **Data Augmentation**: Advanced augmentation techniques for remote sensing
- [ ] **Vocabulary Expansion**: Include more descriptive terms and technical vocabulary
- [ ] **Cross-Domain Transfer**: Adapt model for different types of remote sensing imagery
- [ ] **Real-time Inference**: Optimize model for deployment and real-time applications
- [ ] **Attention Visualization**: Enhanced attention mechanism visualization tools

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

*This implementation serves as a comprehensive example of image captioning for remote sensing imagery, demonstrating both traditional LSTM-based and modern Transformer-based approaches with thorough analysis and comparison of different architectures.*
