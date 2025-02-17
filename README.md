# C.H.O.N.K. 🐱🤖  
**Convolutional Heftiness Observer for Notable Kitties**  

A deep learning project that uses computer vision to classify cats based on their "chonkiness" level. Built with TensorFlow and ResNet50, C.H.O.N.K. can determine if a cat is:  
- **Normal** (regular, skinny, or small cats)  
- **Chonky** (overweight or chunky cats)  

## 🚀 Quick Start  

install requirements.txt
run crawl.py to download images
run preprocess.py to preprocess the images
run train.py to train the model
run evaluate.py to evaluate the model
run python3 predict.py input.img to predict on new images


## 🚀 Project Overview  
1. **Data Collection** – Automatically scrapes and downloads images of fat cats.  
2. **Data Preprocessing** – Resizes, normalizes, and splits the dataset.  
3. **Model Training** – Uses a CNN (e.g., ResNet, VGG16) to classify cats.  
4. **Evaluation** – Tests accuracy, precision, recall, and confusion matrix.  
5. **(Optional) Deployment** – Web or mobile app to classify a user's cat.  

## 📁 Project Structure
- `crawl.py` - Downloads cat images from the web
- `preprocess.py` - Processes and prepares images for training
- `train.py` - Trains the CNN model
- `evaluate.py` - Generates performance metrics and visualizations
- `predict.py` - Makes predictions on new images
- `models/` - Stores trained models
- `processed_data/` - Stores preprocessed image data
- `evaluation/` - Stores evaluation results

## 📈 Model Architecture
- Base: ResNet50 (pretrained on ImageNet)
- Custom top layers with:
  - Global Average Pooling
  - Dense layers (2048, 1024, 512)
  - Batch Normalization
  - Dropout for regularization
  - Binary classification output

## 🎯 Performance
Check `evaluation/` folder after training for:
- Classification Report
- Confusion Matrix
- ROC Curve

## 📝 License
MIT License

## 🤝 Contributing
Contributions welcome! Please feel free to submit a Pull Request.
