![IQRG Banner for Research Projects](../IQRG_Banner_Research_Projects_2024.png)

# Predicting Best-Selling Mobile Phones

This project uses classical and quantum machine learning to predict mobile phone sales based on the Kaggle dataset (https://www.kaggle.com/datasets/muhammedtausif/best-selling-mobile-phones)

## Technologies Used

### Classical Machine Learning

- **Linear Regression**: Predicts sales based on features such as manufacturer and technical specifications.
- **Data Preprocessing**: Utilizes techniques like one-hot encoding for the manufacturer and standardization for numerical features (units sold).
- **Evaluation**: Metrics such as Mean Squared Error (MSE) and R-squared are used to assess model performance.
- **Visualization**: Plots relationships between variables and compares actual vs. predicted sales.

### Quantum Machine Learning

- **Quantum Neural Networks (QNNs)**: Implemented using PennyLane, QNNs use variational quantum circuits to learn from data.
- **Feature Encoding**: Classical data is transformed into quantum states suitable for processing on quantum devices.
- **Entanglement Layers**: Basic entangler layers enhance model complexity and capture intricate relationships among features.
- **Interface**: Integration with TensorFlow allows seamless training and evaluation using quantum circuits.

## Workflow

1. **Data Collection**: Gathered from e-commerce platforms, manufacturer websites, and market reports to capture mobile phone specifications and user sentiment.

2. **Feature Engineering**: Extracted technical specifications, processed user reviews, and incorporated market trends for comprehensive feature sets.

3. **Model Training**: Classical models (e.g., Linear Regression) and quantum models (e.g., QNNs) are trained on processed data to predict sales.

4. **Evaluation and Validation**: Performance metrics validate model accuracy and robustness, ensuring reliable predictions for stakeholders.

5. **Visualization**: Graphical representations aid in understanding relationships between input features and predicted sales outcomes.

## Outcome

This project aims to provide valuable insights to mobile phone manufacturers, retailers, and industry stakeholders for optimizing product development and marketing strategies based on advanced machine learning techniques.

> The research poster can be found in the [IQRG Proceedings 2024](https://thinkingbeyond.education/iqrg_proceedings_2024/)
