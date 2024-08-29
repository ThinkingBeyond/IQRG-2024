![IQRG Banner for Research Projects](../IQRG_Banner_Research_Projects_2024.png)

# Spotify Music Recommendation System using Quantum Machine Learning

Spotify, the world's largest on-demand music service, is best known for its user experience, music recommendation that is constantly getting improved.

Objective: This project aims to apply Quantum Machine Learning (QML) techniques to predict Spotify user preferences. Leveraging quantum computing can potentially enhance the efficiency and accuracy of machine learning models in handling large and complex Spotify music and users datasets.

Dataset : In this project, a dataset comprising 1200 songs mapped to users' personal traits was utilized. The file includes features of Spotify tracks along with corresponding labels that indicate user preferences.

Data Preparation : Features from the dataset are standardized and reduced in dimensionality using PCA. The features are scaled to fit the range ([-π, π]) for compatibility with quantum circuits.

Preprocessing:
The input data (features of songs) is cleaned and standardized.
Dimensionality reduction (PCA) ensures that the data is suitable for quantum processing.
The data is scaled to the appropriate range for quantum gate rotations.

Quantum Circuit Encoding:
Each song's features are converted into a quantum circuit, where each feature controls the rotation of a qubit.

Quantum Machine Learning Algorithm used:
The code uses a hybrid quantum-classical machine learning approach, specifically leveraging quantum circuits within a neural network framework. This involves the following components:

Quantum Circuits: Each data sample is encoded into a quantum circuit.

Parameterized Quantum Circuits (PQCs): These circuits have trainable parameters that can be optimized during the training process.
Quantum Layers in TensorFlow Quantum: The quantum circuits are integrated into a TensorFlow Keras model using TensorFlow Quantum, which allows quantum circuits to be used as layers in a neural network.

Quantum Model Training:
The encoded circuits are input into a PQC within a neural network.
The model learns to distinguish between songs that should be recommended and those that should not based on training labels.

Prediction:
For new or test songs, the same preprocessing and encoding steps are applied.
The trained quantum model predicts the recommendation score for each song.

The output from the prediction model shows the predicted values for each input data point, ranging from negative to positive values. The predictions output array contains continuous values which represent the model's recommendations for the test data:
Positive Values: High positive values close to 1 indicate a strong recommendation.
Negative Values: Values closer to -1 indicate weak or no recommendation.
Thresholding: To make practical recommendations, the model can filter predictions by applying a threshold (e.g., only consider predictions above 0.8 as strong recommendations).

Recommendation:
The predicted scores are analyzed.
Songs with high positive scores are considered strong recommendations.
These songs can then be suggested to the user based on their preference and listening history

> The research poster can be found in the [IQRG Proceedings 2024](https://thinkingbeyond.education/iqrg_proceedings_2024/)
> 
References:
Hurtado, A., Wagner, M. and Mundada, S., 2019. Thank you, Next: Using NLP Techniques to Predict Song Skips on Spotify based on Sequential User and Acoustic Data.
Gori, M., & Pucci, R. (2020). Quantum Algorithms for Recommender Systems. IEEE Transactions on Quantum Engineering.
Tang, E., 2019, June. A quantum-inspired classical algorithm for recommendation systems. ACM SIGACT symposium on theory of computing (pp. 217-228).
Pilato, G. and Vella, F., 2022. A survey on quantum computing for recommendation systems. Information, 14(1), p.20.
C.W. Chen, P. Lamere, M. Schedl, and H. Zamani. Recsys Challenge 2018: Automatic Music Playlist Continuation. 2th ACM Conference on Recommender Systems (RecSys ’18), 2018.
https://quantumai.google/cirq/start/basics
https://www.tensorflow.org/quantum/concepts

