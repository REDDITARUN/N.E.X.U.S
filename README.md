# 🧠 Deep Learning Architectures: CNN, RNN, LSTM, Autoencoder, and Transformer

## Team Members

-   **Charvi Kusuma** [GitHub](https://github.com/kcharvi)
-   **Tarun Reddi** [GitHub](https://github.com/REDDITARUN)

## 📋 Overview

This repository contains implementations of various deep learning architectures including Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), Long Short-Term Memory networks (LSTM), Autoencoders, and Transformers. These models are designed for a range of tasks from image classification to time-series forecasting and anomaly detection.

## 🌟 Importance of This Work

Deep learning has revolutionized many fields by enabling the development of models that can learn complex patterns from data. Here’s why each of these architectures is significant:

- **🖼️ CNNs** excel at capturing spatial hierarchies in images, making them ideal for image classification and object detection tasks.
- **⏳ RNNs** are designed to handle sequential data, making them suitable for tasks such as time-series forecasting and natural language processing.
- **🔁 LSTMs** improve upon RNNs by addressing the vanishing gradient problem, making them more effective at learning long-term dependencies.
- **🔍 Autoencoders** are powerful for unsupervised learning tasks such as anomaly detection, by learning efficient representations of input data.
- **🚀 Transformers** have become the state-of-the-art for many NLP tasks due to their ability to handle long-range dependencies with self-attention mechanisms.

## 📊 Data

We use a variety of datasets to demonstrate the capabilities of each architecture:

1. **🖼️ CNN**: Image classification dataset with 30,000 samples divided into three classes. Each image is 64x64 pixels.
2. **📈 RNN and LSTM**: Time-series data from the Numenta Anomaly Benchmark (NAB), focusing on Apple data from Twitter mentions.
3. **🔧 Autoencoder**: Sensor data from a manufacturing process for anomaly detection.
4. **📰 Transformer**: Text classification using the AG News dataset.

## 🏗️ Model Implementations

### 1. 🖼️ Convolutional Neural Network (CNN)

- **Architecture**: VGG13 with convolutional layers followed by fully connected layers.
- **Techniques Applied**: Regularization, Dropout, Early Stopping, and Data Augmentation.
- **Performance**: Achieved a test accuracy of 91.83%, precision of 90.63%, recall of 90.22%, and F1 score of 90.14%.

### 2. ⏳ Recurrent Neural Network (RNN) and LSTM

- **RNN Architecture**: Vanilla RNN and LSTM models for time-series forecasting.
- **Techniques Applied**: Hyperparameter tuning, optimizer variations (SGD, RMSProp).
- **Performance**: LSTM model achieved better stability and lower error rates compared to vanilla RNNs.

### 3. 🔍 Autoencoders

- **Architecture**: Deep autoencoder with fully connected layers for anomaly detection in manufacturing.
- **Techniques Applied**: Different activation functions, LSTM layers for sequence data.
- **Performance**: Successfully detected anomalies with a low testing loss, indicating good model fit.

### 4. 🚀 Transformers

- **Architecture**: Simplified Transformer with self-attention and feed-forward layers for text classification.
- **Techniques Applied**: Dropout, Early Stopping, Regularization.
- **Performance**: The Dropout + Early Stopping model achieved the highest testing accuracy and lowest testing loss, outperforming other configurations.

## 📈 Results and Discussion

- **🖼️ CNN**: Data augmentation significantly improved generalization, achieving the highest test accuracy among all techniques applied.
- **⏳ RNN and LSTM**: LSTM models outperformed vanilla RNNs, especially in handling long-term dependencies in time-series data.
- **🔍 Autoencoders**: Effective in anomaly detection, with LSTM-based autoencoders providing robust performance on sequence data.
- **🚀 Transformers**: Showed high accuracy in text classification tasks, with the Dropout + Early Stopping model demonstrating the best balance between training accuracy and generalization.

## 🏁 Conclusion

This repository showcases the versatility and power of different deep learning architectures across various domains. By applying techniques such as regularization, dropout, and early stopping, we demonstrate how to enhance model performance and generalization.

## 📚 References

- Illustrated Self-Attention: [Illustrated Self-Attention](https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a)
- Time-Series Data Visualization: [Time-Series Data Visualization](https://machinelearningmastery.com/time-series-data-visualization-with-python/)
- Predicting Stock Prices Using LSTM: [Predicting Stock Prices Using LSTM](https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233)

---
### 🚨Academic Integrity Disclaimer🚨

This project in this repository is intended solely as an inspiration for your future projects and should be referenced accordingly. It is not meant for students to fulfill their academic project requirements. If a student uses this project for such purposes, the creators are not responsible. The student will be solely accountable for violating academic integrity. We explicitly state that this repository should not be used to meet academic requirements. Therefore, any academic integrity issues should be addressed with the student, not the creators.

This work was completed as part of a course at the University at Buffalo. This report is intended to showcase our work for the purpose of sharing knowledge, not for any student to use it to satisfy academic requirements.

This repo file provides a detailed overview of the models, their significance, data used, performance metrics, and key findings from the experiments conducted.
