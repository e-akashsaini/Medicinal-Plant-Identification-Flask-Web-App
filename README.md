# Medicinal Plant Identification Flask Web App

This project presents a Flask-based web application for identifying medicinal plants in India. By utilizing image processing, machine learning techniques, and deep learning models (CNN and DenseNet121), the app allows users to upload photos of medicinal plants for classification.

## Project Overview

This project addresses critical challenges in identifying medicinal plants in India, crucial for Ayurvedic pharmaceutics. It provides a web-based solution where users can upload images of plants, and the application predicts the plant class based on the uploaded image using machine learning models.

The app leverages a Kaggle dataset featuring 40 classes of medicinal plants and implements Convolutional Neural Networks (CNN) and transfer learning with DenseNet121 to perform plant classification.

### Key Achievements:
- **CNN Model Accuracy**: 69.58%
- **DenseNet121 Model Accuracy**: 77.20%

## Features
- Users can upload plant images through a simple web interface.
- The app predicts plant species using machine learning models.
- Supports two types of models: a custom CNN and a DenseNet121 model for transfer learning.
- Displays the model prediction with a confidence score.

## Technologies Used
- **Flask**: Web framework for Python
- **TensorFlow/Keras**: For training and implementing deep learning models
- **OpenCV**: Image processing
- **HTML/CSS**: Front-end design
- **Docker (optional)**: For containerization

## Models
1. **CNN Model**: A custom-built Convolutional Neural Network trained from scratch to classify medicinal plants.
2. **DenseNet121**: A transfer learning model pre-trained on the ImageNet dataset, fine-tuned for plant classification.

### Model Performance
- **CNN Model**: 69.58% accuracy
- **DenseNet121 Model**: 77.20% accuracy
- **Metrics Evaluated**: Precision, Recall, F1-score

## Dataset

The dataset used in this project is sourced from Kaggle and includes images of 40 different medicinal plant species. You can download the dataset from [here](https://www.kaggle.com/datasets) and store it in the `dataset/` folder.

### Prerequisites
- Python 3.8 or higher
- Virtual environment (optional but recommended)
- Flask
- TensorFlow
- OpenCV

### Steps to Install
1. Clone the repository:
   git clone https://github.com/your-username/medicinal-plant-classifier.git
   cd medicinal-plant-classifier
2. pip install -r requirements.txt
3. Download the dataset from Kaggle and place it in the dataset/ folder.
4. python app.py
5. Open your browser and go to http://127.0.0.1:5000/ to access the app.

### Usage
On the homepage, upload an image of a medicinal plant using the upload button.
Once the image is uploaded, the app will process it and provide a prediction along with the confidence score.
The result page will display the plant species predicted by the model.

### Future Work
**Dataset Expansion:** Incorporating more plant species and larger datasets.
**Real-Time Detection:** Implementing real-time plant identification via mobile applications.
**Environmental Data:** Integrating additional environmental parameters to enhance predictions.
**Model Optimization:** Continuous improvement of model accuracy and performance.
**Global Collaboration:** Partnering with researchers and botanists for knowledge sharing and validation.

### Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.
