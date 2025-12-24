# AI Text Detection: Binary and Multiclass Classification

A comprehensive deep learning project for detecting AI-generated text using transformer-based models. This repository contains two Jupyter notebooks that implement binary classification (AI vs Human) and multiclass classification (identifying specific AI models) for text detection.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Notebooks](#notebooks)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Datasets](#datasets)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🔍 Overview

This project tackles the critical challenge of distinguishing AI-generated text from human-written content. With the proliferation of large language models (LLMs) like GPT-4, ChatGPT, Claude, and others, identifying the source of text content has become increasingly important for academic integrity, content authenticity, and misinformation prevention.

The repository provides two complementary approaches:
1. **Binary Classification**: Distinguishes between AI-generated and human-written text
2. **Multiclass Classification**: Identifies the specific AI model that generated the text

## ✨ Features

- **Transformer-Based Architecture**: Leverages pre-trained transformer models (BERT, RoBERTa, etc.) for state-of-the-art performance
- **Comprehensive Feature Engineering**: 
  - Linguistic features (POS tags, sentence structure)
  - Statistical features (text complexity, diversity metrics)
  - Transformer embeddings
- **Advanced Model Components**:
  - Multi-head attention mechanisms
  - Feature fusion layers
  - Dropout and regularization techniques
- **Extensive Evaluation**:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrices
  - ROC curves and AUC scores
  - Per-class performance metrics
- **Visualization Tools**: 
  - Training/validation curves
  - Feature importance analysis
  - Performance comparison plots
- **Robust Data Processing**:
  - Text preprocessing and normalization
  - Feature caching for efficiency
  - Batch processing support

## 📓 Notebooks

### 1. Binary AI Text Detection (`binary-ai-mega-data_text_detection.ipynb`)

**Purpose**: Binary classification to distinguish AI-generated text from human-written content.

**Key Components**:
- BERT-based feature extraction
- Custom neural network architecture
- Comprehensive evaluation metrics
- Visualization of results

**Dataset**: AI Mega Data dataset with binary labels (AI/Human)

### 2. Multiclass Text Detection (`Multiclass_text_detection_raid_dataset.ipynb`)

**Purpose**: Multi-class classification to identify the specific AI model that generated the text.

**Supported Models**:
- GPT-4
- GPT-3
- GPT-2
- ChatGPT
- Claude
- Cohere (& Cohere-Chat)
- MPT (& MPT-Chat)
- Mistral (& Mistral-Chat)
- LLaMA-Chat

**Key Components**:
- 12-class classification system
- Class imbalance handling
- Advanced feature fusion
- Interpretability tools (SHAP, Integrated Gradients)

**Dataset**: RAID (Robust AI Detection) dataset with multi-label annotations

## 🛠️ Requirements

### Core Dependencies
```
Python 3.8+
PyTorch 1.12+
transformers 4.20+
scikit-learn 1.0+
pandas
numpy
matplotlib
seaborn
nltk
tqdm
```

### Optional Dependencies
```
captum (for model interpretability)
shap (for SHAP analysis)
jupyter (for running notebooks)
```

## 📦 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/ai-text-detection.git
cd ai-text-detection
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install torch torchvision torchaudio
pip install transformers scikit-learn pandas numpy matplotlib seaborn nltk tqdm
pip install captum shap jupyter
```

4. **Download NLTK data**:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
```

## 🚀 Usage

### Running the Notebooks

1. **Start Jupyter Notebook**:
```bash
jupyter notebook
```

2. **Open the desired notebook**:
   - For binary classification: `binary-ai-mega-data_text_detection.ipynb`
   - For multiclass classification: `Multiclass_text_detection_raid_dataset.ipynb`

3. **Run cells sequentially** to:
   - Load and preprocess data
   - Train the model
   - Evaluate performance
   - Visualize results

### Quick Start Example

```python
# Binary Classification Example
from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained model
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = YourTrainedModel()  # Load your trained model

# Predict
text = "Your text here..."
inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1)
print("AI-generated" if prediction == 1 else "Human-written")
```

## 🏗️ Model Architecture

### Binary Classification Model
- **Base Model**: BERT/RoBERTa transformer
- **Feature Layers**: 
  - Transformer embeddings (768-dim)
  - Linguistic features
  - Statistical features
- **Classification Head**: 
  - Multi-layer perceptron
  - Dropout regularization
  - Sigmoid/Softmax output

### Multiclass Classification Model
- **Base Model**: Transformer encoder
- **Feature Fusion**: Combines multiple feature types
- **Multi-head Attention**: Captures complex patterns
- **Output Layer**: 12-class softmax

## 📊 Datasets

### Binary Classification Dataset
- **Source**: AI Mega Data
- **Size**: Large-scale corpus
- **Labels**: AI-generated, Human-written
- **Content**: Various text types and domains

### Multiclass Classification Dataset
- **Source**: RAID (Robust AI Detection) Dataset
- **Size**: Comprehensive multi-model collection
- **Labels**: 12 different AI models + human
- **Features**: Title and generation text pairs

## 📈 Results

### Binary Classification
- **Accuracy**: 98.10%
- **Precision**: 97.39%
- **Recall**: 88.65%
- **F1-Score**: 96.3%

### Multiclass Classification
- **Overall Accuracy**: 88.34%
- **Per-class F1**: Varies by model
- **Confusion Matrix**: Detailed in notebook

*Note: Run the notebooks to see actual performance metrics with your dataset*

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Areas for Contribution
- Additional datasets
- New model architectures
- Performance optimizations
- Documentation improvements
- Bug fixes

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace Transformers library
- PyTorch team
- NLTK contributors
- Dataset creators and maintainers

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🔗 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

---

**Star ⭐ this repository if you find it helpful!**
