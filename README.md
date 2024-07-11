# Optimized Transformers for Precise Data Insights


## Team Members

-   **Charvi Kusuma** [GitHub](https://github.com/kcharvi)
-   **Tarun Reddi** [GitHub](https://github.com/REDDITARUN)

## 📋 Overview

This repository contains implementations of various deep learning architectures including Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), Long Short-Term Memory networks (LSTM), Autoencoders, and Transformers. These models are designed for a range of tasks from image classification to time-series forecasting and anomaly detection.

## Introduction

In recent years, advancements in machine learning and deep learning have led to the creation of powerful models that excel in various tasks. This document summarizes three projects that tackle different challenges: anomaly detection, text classification, and image classification. 

The first project uses autoencoders to detect anomalies in the Twitter volume of Apple Inc. (AAPL) over time. The second project involves building a transformer model in PyTorch for text classification, focusing on data preprocessing and optimization. The final project explores Vision Transformer (ViT) and EfficientNet models for image classification, showcasing their unique architectures and effectiveness.

These projects highlight the flexibility and capability of modern neural networks in solving complex, data-driven problems.

## Autoencoders for Anomaly Detection

This project focuses on utilizing autoencoders for anomaly detection in time series data. The dataset in use is the Twitter volume of Apple Inc. (AAPL) over time. The goal is to identify anomalies within this time series data using various versions of autoencoder models.

**Dataset:**
- **Source:** Twitter volume data for Apple Inc. (AAPL).
- **Data Description:** The dataset consists of timestamps and corresponding Twitter volume values.
- **Preprocessing:**
  - The timestamps were converted to datetime format.
  - Additional features such as month, day, hour, and day of the year were extracted from the timestamp.
  - Data was normalized using MinMaxScaler to scale the 'value' column between 0 and 1.

**Data Visualization:**
1. **Time Series Plot:** Displayed the entire series to visualize trends and potential anomalies.
2. **Seasonal Plot:** Examined monthly trends to identify seasonality.
3. **Seasonal Polar Plot:** Created a polar plot to better understand the seasonal patterns over a year.

![Time Series Plot](/assets/time_series_plot.png) 
![Seasonal Plot](/assets/seasonal_plot.png)
![Seasonal Polar Plot](/assets/seasonal_polar_plot.png)

**Data Splitting:**
- The dataset was split into training, validation, and test sets:
  - **Training Set:** 70%
  - **Validation Set:** 15%
  - **Test Set:** 15%

**Model Development:**
1. **Base Model:**
   - A basic autoencoder model was constructed as a starting point.
   - Architecture: Simple feedforward neural network with ReLU activation functions.

2. **Version 1 - Increased Number of Layers:**
   - Enhanced the base model by increasing the number of hidden layers to capture more complex patterns in the data.

3. **Version 2 - Mixed Activation Functions:**
   - Experimented with different activation functions (ReLU, Tanh, Sigmoid) within the network to observe their impact on the model's performance.

4. **Version 3 - Incorporation of LSTM:**
   - Given the time series nature of the data, an LSTM-based autoencoder was developed to leverage temporal dependencies and improve anomaly detection accuracy.

**Results:**
- **Base Model:** Provided a baseline for comparison but struggled with capturing more nuanced anomalies.
- **Version 1:** Showed improvement with deeper architecture but faced issues of overfitting.
- **Version 2:** Mixed activations yielded diverse results; some configurations improved generalization while others did not.
- **Version 3:** The LSTM autoencoder demonstrated the best performance, effectively capturing temporal patterns and detecting anomalies more accurately.

![Base Model Detections](/assets/autoencoder_base.png)

![Improved Model Detections](/assets/autoencoder_improved.png)

The project highlights the evolution of model complexity and its impact on anomaly detection performance in time series data. The LSTM-based autoencoder emerged as the most effective approach, demonstrating the importance of leveraging temporal dependencies for anomaly detection in sequential data.

<hr>

## Transformers for Text Classification

Here we involve with building and training a transformer model using PyTorch for text classification tasks. The aim is to preprocess the text data, construct an effective transformer model, and optimize it for better performance.

**Dataset:**
- **Source:** The dataset used is specified as `train.csv` and contains text data with labels.
- **Data Description:** The dataset includes text entries and corresponding classification labels.
- **Preprocessing:**
  - Basic data exploration to understand the distribution and statistics of the dataset.
  - Text cleaning: Removal of punctuation, stop words, and unnecessary characters.
  - Lowercasing: Ensuring all text is in lowercase for consistent representation.
  - Tokenization: Breaking down text into individual words using libraries like NLTK or SpaCy.
  - Vocabulary building: Creating a vocabulary of unique tokens from the dataset.
  - Numerical representation: Converting tokens to numerical representations using techniques like Word2Vec or GloVe.

**Data Visualization:**
1. **Average Word Count by Category:** A bar chart showing the average word count per category to identify differences in text length across categories.
2. **Word Cloud:** A word cloud visualization to display the most frequent words in the text data, providing insights into common terms and potential stop words.
3. **Additional Visualization:** Potentially more visualizations, such as polarity distribution or word count distribution, to further understand the text characteristics.

![Word Cloud](/assets/transformer_word_cloud.png)
![Cleaned Data](/assets/transformers_cleaned_dataset.png)

**Model Development:**
1. **Model Construction:**
   - A transformer model was built using PyTorch, focusing on leveraging the attention mechanism for effective text classification.
   - Architecture: Consisted of embedding layers, attention mechanisms, and feedforward neural networks.

2. **Training the Transformer:**
   - The model was trained on the preprocessed and tokenized text data.
   - Techniques like early stopping and dropout were employed to prevent overfitting and enhance generalization.

3. **Evaluation and Optimization:**
   - The model's performance was evaluated using metrics such as precision, recall, F1-score, and confusion matrices.
   - Regularization techniques, including L2 regularization, were applied to improve the model's robustness.

For easier understanding of how to get started, consider the below snippet:

**TransformerBlock Class:**
This class defines a transformer block, which is a key building block of the transformer model. It includes multi-head self-attention and a feedforward neural network, along with layer normalization to stabilize training.

```python
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)  # Self-attention mechanism
        out1 = self.layernorm1(inputs + attn_output)  # Add & Normalize
        ffn_output = self.ffn(out1)  # Feedforward network
        return self.layernorm2(out1 + ffn_output)  # Add & Normalize
```
- **MultiHeadAttention:** This layer performs the attention mechanism, allowing the model to focus on different parts of the input sequence.
- **Feedforward Network:** A small neural network that processes the output from the attention layer.
- **LayerNormalization:** Normalizes the outputs to improve training stability.

**TokenAndPositionEmbedding Class:**
This class handles token and position embeddings, which are crucial for transformers to understand the order and meaning of tokens in the input sequence.

```python
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
```
- **Embedding:** Converts input tokens into dense vectors of fixed size.
- **Position Embedding:** Adds positional information to the token embeddings, helping the model understand the order of tokens in the sequence.

The `TransformerBlock` focuses on attention and transformation of token embeddings, while the `TokenAndPositionEmbedding` provides meaningful vector representations of tokens and their positions in the input sequence. These components together enable the transformer model to effectively process and understand text data for classification tasks.

**Optimization Techniques:**
- **Dropout:** Applied dropout layers to prevent overfitting by randomly setting a fraction of input units to zero during training.
- **Early Stopping:** Implemented early stopping to halt training when the model's performance on the validation set stops improving, thus avoiding overfitting.
- **Regularization:** Used L2 regularization to penalize large weights, encouraging the model to find simpler and more generalizable patterns.

**Results:**
- The transformer model demonstrated significant improvements in text classification tasks compared to baseline models.
- Visualizations and preprocessing steps provided valuable insights into the dataset and informed model construction.
- Optimization techniques like dropout, early stopping, and regularization effectively enhanced the model's performance and prevented overfitting.

The complete performance results can be seen below:
![Results](/assets/transformers_results.png)

The project showcases the power of transformer models for text classification, emphasizing the importance of thorough data preprocessing, visualization, and optimization. The transformer model, with its attention mechanism, effectively captured complex patterns in the text data, leading to superior classification performance. This approach provides a robust framework for tackling various text classification challenges in natural language processing (NLP).

<hr>

## Vision Transformer (ViT) and EfficientNet for Image Classification

This project aims to explore the performance of Vision Transformer (ViT) and EfficientNet models for image classification tasks. Both models represent state-of-the-art architectures in the field of deep learning for computer vision.

**Dataset:**
- **Source:** A custom image dataset located in a zipped file (`cnn_dataset.zip`).
- **Data Description:** The dataset consists of images organized into categories, each representing a different class.
- **Preprocessing:**
  - Images were loaded and resized to a consistent shape (64x64 pixels).
  - Pixel values were normalized, and statistical measures such as mean and standard deviation of pixel values were computed.

**Data Statistics:**
- **Number of Examples:** The dataset includes multiple examples, but the exact number depends on the size of the categories.
- **Number of Categories:** Various categories representing different classes are present in the dataset.
- **Image Shape:** Images were standardized to 64x64 pixels.
- **Mean Pixel Value:** The average pixel value across the dataset was computed.
- **Standard Deviation of Pixel Values:** The standard deviation of pixel values was also calculated.

**Model Development:**

1. **Vision Transformer (ViT):**
   - **Architecture:** ViT applies transformer models, which have been successful in NLP tasks, to image classification. It divides images into patches and treats them as sequences, similar to words in a sentence.
   - **Components:** Key components include embedding layers, multi-head self-attention, and feedforward neural networks.
   - **Training:** The ViT model was trained on the preprocessed images, leveraging its ability to capture global dependencies through self-attention mechanisms.

### Explanation of Vision Transformer (ViT) Implementation

**Overview:**
The Vision Transformer (ViT) is a model that adapts the transformer architecture, originally designed for natural language processing, to the domain of image classification. The core idea is to split images into patches, embed these patches, and then process them using transformer layers.

**Components:**

1. **Patches Layer:**
   - **Purpose:** This layer divides the input image into smaller patches.
   - **Implementation:**
     ```python
     class Patches(tf.keras.layers.Layer):
         def __init__(self, patch_size, **kwargs):
             super(Patches, self).__init__(**kwargs)
             self.patch_size = patch_size

         def call(self, images):
             batch_size = tf.shape(images)[0]
             patches = tf.image.extract_patches(
                 images=images,
                 sizes=[1, self.patch_size, self.patch_size, 1],
                 strides=[1, self.patch_size, self.patch_size, 1],
                 rates=[1, 1, 1, 1],
                 padding='VALID',
             )
             patch_dims = patches.shape[-1]
             patches = tf.reshape(patches, [batch_size, -1, patch_dims])
             return patches
     ```
   - **Explanation:** The `call` method extracts patches from the input image and reshapes them into a suitable format for further processing.

2. **PatchEncoder Layer:**
   - **Purpose:** This layer projects the patches into a higher-dimensional space and adds positional embeddings to maintain spatial information.
   - **Implementation:**
     ```python
     class PatchEncoder(tf.keras.layers.Layer):
         def __init__(self, num_patches, projection_dim, **kwargs):
             super(PatchEncoder, self).__init__(**kwargs)
             self.num_patches = num_patches
             self.projection_dim = projection_dim
             self.position_embedding = tf.keras.layers.Embedding(
                 input_dim=num_patches, output_dim=projection_dim
             )

         def build(self, input_shape):
             self.projection = Dense(units=self.projection_dim)

         def call(self, patch):
             positions = tf.range(start=0, limit=self.num_patches, delta=1)
             encoded = self.projection(patch) + self.position_embedding(positions)
             return encoded
     ```
   - **Explanation:** The `PatchEncoder` class includes a projection layer to embed the patches and an embedding layer to add positional information.

3. **ViT Model Creation:**
   - **Purpose:** This function builds the complete Vision Transformer model.
   - **Implementation:**
     ```python
     def create_vit_classifier():
         input_shape = (64, 64, 3)
         num_classes = len(categories)
         inputs = Input(shape=input_shape)
         rescaled_inputs = Rescaling(scale=1./255)(inputs)

         patch_size = 8
         num_patches = (64 // patch_size) ** 2
         patch_dim = 3 * patch_size ** 2

         patches = Patches(patch_size)(rescaled_inputs)
         encoded_patches = PatchEncoder(num_patches, patch_dim)(patches)

         for _ in range(4):  # Let's create 4 layers of Transformer encoders
             x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
             attention_output = tf.keras.layers.MultiHeadAttention(
                 num_heads=4, key_dim=patch_dim, dropout=0.1
             )(x1, x1)
             x2 = Add()([attention_output, encoded_patches])
             x3 = LayerNormalization(epsilon=1e-6)(x2)
             x3 = Dense(units=2048, activation='relu')(x3)
             x3 = Dense(units=patch_dim)(x3)
             encoded_patches = Add()([x3, x2])

         representation = Flatten()(encoded_patches)
         representation = Dense(units=2048, activation='relu')(representation)
         representation = Dropout(0.5)(representation)
         outputs = Dense(units=num_classes, activation='softmax')(representation)
         model = Model(inputs=inputs, outputs=outputs)

         return model
     ```
   - **Explanation:**
     - **Input Processing:** The input images are rescaled to the range [0, 1].
     - **Patch Extraction:** Images are divided into patches of size 8x8.
     - **Patch Encoding:** Each patch is embedded and enriched with positional information.
     - **Transformer Layers:** Four layers of transformer encoders are used to process the patches. Each encoder consists of:
       - **LayerNormalization:** Normalizes the input.
       - **MultiHeadAttention:** Computes attention over the patches.
       - **Residual Connections:** Adds the input to the output of the attention layer and the feedforward network to form residual connections.
       - **Feedforward Network:** Further processes the data using dense layers.
     - **Final Representation:** The encoded patches are flattened and passed through dense layers with dropout for regularization.
     - **Output Layer:** A softmax layer is used for classification, outputting the probability distribution over the classes.

The Vision Transformer model thus leverages the power of self-attention to process images, capturing global relationships and dependencies between different parts of the image, leading to robust image classification performance.

2. **EfficientNet:**
   - **Architecture:** EfficientNet uses a compound scaling method that uniformly scales network width, depth, and resolution using a set of fixed scaling coefficients.
   - **Components:** It includes layers such as convolutional layers, batch normalization, activation functions, and global average pooling.
   - **Training:** The EfficientNet model was trained on the same preprocessed dataset, utilizing its efficiency and accuracy for image classification tasks.

**Model Training and Evaluation:**
- **Training:** Both models were trained using techniques like data augmentation, early stopping, and dropout to improve generalization and prevent overfitting.
- **Evaluation Metrics:** Performance was evaluated using metrics such as accuracy, precision, recall, F1-score, and confusion matrix.

**Results:**
- **ViT Model:** The ViT model showcased its strength in capturing global patterns and dependencies in the images, resulting in high classification accuracy.
- **EfficientNet Model:** EfficientNet demonstrated its efficiency and scalability, achieving competitive performance with fewer computational resources compared to traditional CNNs.

![ViT Transformer Results](/assets/vision_transformer_results.png)

Efficient Net Results:

![EfficientNet Results](/assets/efficientnet_results.png)

The project highlights the effectiveness of both Vision Transformer and EfficientNet models for image classification tasks. ViT's attention mechanism allows it to capture intricate patterns in images, while EfficientNet's compound scaling approach ensures high performance with optimized resource utilization. This study underscores the importance of choosing the right model architecture based on the specific requirements and constraints of the image classification task at hand.

<hr>

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
