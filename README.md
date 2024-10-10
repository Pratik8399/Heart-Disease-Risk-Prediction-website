# Heart Disease Risk Prediction Website

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Model Details](#model-details)
8. [Dataset Information](#dataset-information)
9. [Contributing](#contributing)
10. [License](#license)
11. [Contact](#contact)

## Overview

The **Heart Disease Risk Prediction** website allows users to input health-related data and receive a prediction of their heart disease risk. It uses a machine learning model trained on historical health data to provide insights into the likelihood of heart disease based on various factors like age, cholesterol levels, chest pain type, and more.

This project is built using FastAPI for the backend and integrates a pre-trained RandomForestClassifier to make predictions.

## Features

- User-friendly web interface for inputting health data.
- Predicts heart disease risk based on user inputs.
- Real-time predictions using a trained machine learning model.
- RESTful API built with FastAPI for easy integration and scalability.

## Tech Stack

- **Backend**: FastAPI (Python)
- **Machine Learning**: scikit-learn, pandas, NumPy
- **Data Visualization**: Matplotlib, Seaborn (for model evaluation)
- **Model**: RandomForestClassifier
- **Deployment**: Can be deployed on platforms like Heroku, AWS, or any cloud server supporting FastAPI.

## Project Structure

```
Heart-Disease-Risk-Prediction/
│
├── Backend/
│   ├── main.py                 # FastAPI application for backend API
│   ├── code.py                 # Machine learning model training and prediction logic
│   ├── heart (1).csv           # Dataset for training the model
│
├── frontend/                   # (Add details if any frontend code exists)
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation (this file)
└── model/
    └── heart_disease_model.pkl  # Saved model (if applicable)
```

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Pratik8399/Heart-Disease-Risk-Prediction-website.git
    cd Heart-Disease-Risk-Prediction-website
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Run the FastAPI server**:
    ```bash
    uvicorn main:app --reload
    ```

5. **Access the application**:
    - Open your web browser and navigate to `http://localhost:8000/docs` to access the API documentation.

## Usage

1. **Input Data**: Send a POST request to the `/HDP` endpoint with JSON data containing age, chest pain type, gender, and other health metrics.
2. **Get Prediction**: Receive a prediction indicating the likelihood of heart disease based on the provided inputs.
3. **Example Request**:
    ```json
    {
        "age": 45,
        "chestPain": 2,
        "gender": "male",
        "MaxHeartRate": 150,
        "ExerciseInducedAngina": "no",
        "oldpeak": 1.2,
        "slope": 2,
        "vessels": 1,
        "thalassemia": 3
    }
    ```

## Model Details

- **Model Type**: RandomForestClassifier
- **Training Data**: The model was trained on a dataset containing health metrics such as age, gender, chest pain type, resting blood pressure, cholesterol levels, and more.
- **Evaluation Metrics**: Accuracy score, classification report.
- **Preprocessing**: Data was cleaned, normalized, and features were selected to improve model performance.

## Dataset Information

- **Source**: The dataset (`heart (1).csv`) is used to train the heart disease prediction model.
- **Columns**:
    - `age`: Age of the patient.
    - `sex`: Gender (1 = male, 0 = female).
    - `cp`: Chest pain type (0-3).
    - `trestbps`: Resting blood pressure.
    - `chol`: Serum cholesterol in mg/dl.
    - `thalach`: Maximum heart rate achieved.
    - `exang`: Exercise-induced angina (1 = yes, 0 = no).
    - `oldpeak`: ST depression induced by exercise relative to rest.
    - `slope`: Slope of the peak exercise ST segment.
    - `ca`: Number of major vessels colored by fluoroscopy.
    - `thal`: Thalassemia (1 = normal; 2 = fixed defect; 3 = reversible defect).
    - `target`: Diagnosis of heart disease (1 = presence, 0 = absence).

## Contributing

Contributions are welcome! If you would like to improve this project, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions, feel free to reach out:

- **Pratik Dnyaneshwar Kale**
- Email: [kalepratik8399@gmail.com](mailto:kalepratik8399@gmail.com)
- LinkedIn: [linkedin.com/in/pratik-kale32](https://linkedin.com/in/pratik-kale32)
- GitHub: [Pratik8399](https://github.com/Pratik8399)

---
