# 🛡️ End-to-End Intelligent Phishing Defense

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/sanjeevpatil804/End-to-End-intelligent-phishing-defense)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-Deployed-orange.svg)](https://aws.amazon.com/)

A production-ready machine learning system for intelligent phishing website detection using advanced feature engineering, automated hyperparameter optimization, and MLOps best practices. This end-to-end solution leverages 30 carefully engineered features to identify malicious URLs with high accuracy, deployed on AWS with CI/CD automation.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Machine Learning Pipeline](#-machine-learning-pipeline)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Model Performance](#-model-performance)
- [Deployment](#-deployment)
- [CI/CD Pipeline](#-cicd-pipeline)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Project Overview

Phishing attacks remain one of the most prevalent cybersecurity threats, with attackers continuously evolving their tactics to deceive users. This project implements an intelligent, automated defense system that:

- **Detects** phishing websites using machine learning with 30+ URL and content-based features
- **Processes** data through a robust ETL pipeline from MongoDB to trained models
- **Optimizes** model performance using Optuna for automated hyperparameter tuning
- **Deploys** seamlessly via Docker containers on AWS infrastructure
- **Automates** the entire ML lifecycle with CI/CD using GitHub Actions

The system achieves high precision and recall in detecting phishing attempts, making it suitable for integration into web browsers, security platforms, or standalone applications.

---

## ✨ Key Features

### 🔍 **Intelligent Feature Engineering**
- **30 sophisticated features** extracted from URLs and webpage characteristics:
  - IP address detection in URLs
  - URL length analysis
  - URL shortening service detection
  - SSL certificate validation
  - Domain age and registration length
  - Web traffic patterns
  - Google indexing status
  - Statistical report analysis

### 🤖 **Advanced Machine Learning**
- **Automated Model Selection**: Compares Random Forest and XGBoost classifiers
- **Optuna Hyperparameter Tuning**: 50+ trials for optimal parameter discovery
- **Cross-Validation**: Robust 5-fold CV for reliable performance estimation
- **Comprehensive Metrics**: F1-Score, Precision, Recall, and custom classification metrics

### 🏗️ **Production-Ready MLOps**
- **Modular Pipeline Architecture**: Separate components for ingestion, validation, transformation, and training
- **Artifact Management**: Versioned model artifacts with timestamps
- **Model Registry**: Automated model versioning and metadata tracking
- **Exception Handling**: Custom exception framework for robust error management
- **Logging**: Comprehensive logging throughout the pipeline

### 🚀 **Cloud-Native Deployment**
- **FastAPI REST API**: High-performance async API for predictions
- **Docker Containerization**: Consistent deployment across environments
- **AWS ECR Integration**: Automated container registry management
- **GitHub Actions CI/CD**: Automated testing, building, and deployment
- **Scalable Infrastructure**: Self-hosted runner for continuous deployment

---

## 🏛️ Architecture

```
┌─────────────────┐
│   Raw Data      │
│   (MongoDB)     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              Data Ingestion Component                    │
│  • Fetch data from MongoDB                              │
│  • Create train/test split                              │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│           Data Validation Component                      │
│  • Validate schema (30 features)                        │
│  • Check data quality                                   │
│  • Detect data drift                                    │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│        Data Transformation Component                     │
│  • KNN Imputation for missing values                    │
│  • Feature preprocessing pipeline                       │
│  • Save preprocessor object                             │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│            Model Trainer Component                       │
│  • Optuna hyperparameter optimization                   │
│  • Random Forest & XGBoost comparison                   │
│  • Best model selection (F1-Score)                      │
│  • Model evaluation & reporting                         │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│               Prediction Pipeline                        │
│  • Load trained model & preprocessor                    │
│  • Batch prediction API endpoint                        │
│  • HTML table visualization                             │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset

### Features (30 Input Variables)

The model uses 30 carefully engineered features to detect phishing websites:

| Feature Category | Features | Description |
|-----------------|----------|-------------|
| **URL-Based** | `having_IP_Address`, `URL_Length`, `Shortining_Service`, `having_At_Symbol`, `double_slash_redirecting`, `Prefix_Suffix` | Characteristics extracted directly from URL structure |
| **Domain-Based** | `having_Sub_Domain`, `SSLfinal_State`, `Domain_registeration_length`, `age_of_domain`, `DNSRecord` | Domain registration and SSL information |
| **Content-Based** | `Favicon`, `port`, `HTTPS_token`, `Request_URL`, `URL_of_Anchor`, `Links_in_tags`, `SFH`, `Submitting_to_email` | Webpage content analysis |
| **Behavioral** | `Abnormal_URL`, `Redirect`, `on_mouseover`, `RightClick`, `popUpWidnow`, `Iframe` | User interaction and behavioral patterns |
| **Reputation** | `web_traffic`, `Page_Rank`, `Google_Index`, `Links_pointing_to_page`, `Statistical_report` | External reputation metrics |

### Target Variable
- **`Result`**: Binary classification
  - `1`: Legitimate website
  - `-1` / `0`: Phishing website

### Data Source
- **Storage**: MongoDB (Cloud Database)
- **Database**: `SanjeevAI`
- **Collection**: `NetworkData`
- **Format**: CSV → JSON → MongoDB

---

## 🔬 Machine Learning Pipeline

### 1️⃣ **Data Ingestion**
```python
- Connects to MongoDB using PyMongo
- Exports data to DataFrame
- Performs stratified train-test split (80-20)
- Saves raw data artifacts
```

### 2️⃣ **Data Validation**
```python
- Validates schema against data_schema/schema.yaml
- Checks for required 30 features + target
- Ensures data types (all int64)
- Generates validation report
- Detects data drift
```

### 3️⃣ **Data Transformation**
```python
- KNN Imputation for handling missing values
  • n_neighbors: 3
  • weights: 'uniform'
- Separates features and target
- Applies preprocessing pipeline
- Target encoding: -1 → 0 for binary classification
- Saves preprocessor object for inference
```

### 4️⃣ **Model Training with Optuna**
```python
- Automated hyperparameter optimization
  • Study Name: 'phishing_detection_study'
  • Trials: 50
  • Optimization Metric: F1-Score (CV)
  
- Classifiers Compared:
  1. Random Forest
     - n_estimators: [50, 300]
     - max_depth: [5, 30]
     - min_samples_split: [2, 20]
     
  2. XGBoost
     - n_estimators: [50, 300]
     - max_depth: [3, 15]
     - learning_rate: [0.01, 0.3]
     - subsample: [0.6, 1.0]

- Cross-Validation: 5-fold stratified
- Best model selection based on CV F1-score
- Comprehensive evaluation on train/test sets
```

### 5️⃣ **Model Evaluation**
```python
Metrics Tracked:
- F1-Score (Primary metric)
- Precision
- Recall
- Confusion Matrix
- Classification Report

Reports Generated:
- model_evaluation_report.yaml
- Training metrics
- Test metrics
- Best hyperparameters
```

---

## 🛠️ Technology Stack

### **Core Machine Learning**
- **Python 3.10**: Primary programming language
- **scikit-learn**: ML algorithms, preprocessing, metrics
- **XGBoost**: Gradient boosting classifier
- **Optuna**: Automated hyperparameter optimization
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis

### **Data Storage & Management**
- **MongoDB**: Cloud database for raw data storage
- **PyMongo**: Python MongoDB driver
- **PyYAML**: Configuration management

### **Web Framework & API**
- **FastAPI**: Modern, high-performance web framework
- **Uvicorn**: ASGI server for FastAPI
- **Jinja2**: Template engine for HTML rendering
- **Starlette**: ASGI framework components
- **python-multipart**: File upload support

### **Deployment & DevOps**
- **Docker**: Containerization
- **AWS ECR**: Container registry
- **AWS EC2**: Self-hosted runner for deployment
- **GitHub Actions**: CI/CD automation
- **python-dotenv**: Environment variable management
- **certifi**: SSL certificate verification

### **Development Tools**
- **Git**: Version control
- **GitHub**: Code repository and CI/CD
- **VS Code**: Development environment

---

## 📁 Project Structure

```
End-to-End-Network-Security/
│
├── app.py                          # FastAPI application entry point
├── main.py                         # Training pipeline execution script
├── push_data.py                    # MongoDB data ingestion utility
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container configuration
├── Setup.py                        # Package setup configuration
├── .env                            # Environment variables (MongoDB URL)
├── .gitignore                      # Git ignore rules
│
├── .github/
│   └── workflows/
│       └── main.yaml               # CI/CD pipeline configuration
│
├── networksecurity/                # Main package
│   ├── __init__.py
│   │
│   ├── components/                 # Pipeline components
│   │   ├── data_ingestion.py      # MongoDB → CSV extraction
│   │   ├── Data_validation.py     # Schema & quality validation
│   │   ├── data_transformation.py # Preprocessing & imputation
│   │   └── model_trainer.py       # Optuna-based model training
│   │
│   ├── pipeline/                   # End-to-end pipelines
│   │   └── training_pipeline.py   # Complete training workflow
│   │
│   ├── entity/                     # Data classes
│   │   ├── config_entity.py       # Configuration dataclasses
│   │   └── artifact_entity.py     # Artifact dataclasses
│   │
│   ├── constants/                  # Constants & hyperparameters
│   │   └── training_pipeline.py   # Pipeline constants
│   │
│   ├── Exception/                  # Custom exceptions
│   │   └── exception.py            # NetworkSecurityException
│   │
│   └── Utils/                      # Utility functions
│       └── main_utils/
│           ├── utils.py            # General utilities
│           ├── estimator.py        # NetworkModel class
│           ├── classification_metrics.py  # Evaluation metrics
│           └── optuna_tuner.py     # Hyperparameter optimization
│
├── data_schema/
│   └── schema.yaml                 # Feature schema definition
│
├── Network_Data/
│   └── phishingData.csv            # Raw dataset
│
├── artifact/                       # Training artifacts (timestamped)
│   └── [timestamp]/
│       ├── data_ingestion/
│       ├── data_validation/
│       ├── data_transformation/
│       └── model_trainer/
│
├── final_model/                    # Production models
│   ├── model.pkl                   # Trained ML model
│   └── preprocessor.pkl            # Preprocessing pipeline
│
└── prediction_output/              # API prediction results
    └── output.csv
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10+
- MongoDB Atlas account (or local MongoDB)
- Docker (for containerization)
- AWS Account (for deployment)
- Git

### Local Setup

1. **Clone the Repository**
```bash
git clone https://github.com/sanjeevpatil804/End-to-End-intelligent-phishing-defense.git
cd End-to-End-intelligent-phishing-defense
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Environment Variables**

Create a `.env` file in the root directory:
```env
MONGODB_URL_KEY=mongodb+srv://<username>:<password>@cluster.mongodb.net/
```

5. **Push Data to MongoDB** (First Time Only)
```bash
python push_data.py
```

---

## 💻 Usage

### Training the Model

Run the complete training pipeline:

```bash
python main.py
```

This will execute:
1. Data ingestion from MongoDB
2. Data validation against schema
3. Data transformation with KNN imputation
4. Model training with Optuna optimization
5. Model evaluation and artifact generation

**Output:**
- Trained model saved in `artifact/[timestamp]/model_trainer/model.pkl`
- Final model saved in `final_model/model.pkl`
- Preprocessor saved in `final_model/preprocessor.pkl`
- Evaluation report in `artifact/[timestamp]/model_trainer/model_evaluation_report.yaml`

### Starting the API Server

Launch the FastAPI application:

```bash
python app.py
```

The API will be available at:
- **Swagger UI**: `http://localhost:8080/docs`
- **ReDoc**: `http://localhost:8080/redoc`

### Making Predictions

#### Via API (curl)
```bash
curl -X POST "http://localhost:8080/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_data.csv"
```

#### Via Python
```python
import requests

url = "http://localhost:8080/predict"
files = {"file": open("test_data.csv", "rb")}
response = requests.post(url, files=files)
print(response.text)
```

#### Via Swagger UI
1. Navigate to `http://localhost:8080/docs`
2. Click on `/predict` endpoint
3. Click "Try it out"
4. Upload CSV file with 30 features
5. Execute and view results

**Input CSV Format:**
```csv
having_IP_Address,URL_Length,Shortining_Service,...,Statistical_report
-1,54,1,...,1
1,20,-1,...,-1
```

**Output:**
- HTML table with predictions
- CSV file saved in `prediction_output/output.csv`
- `predicted_column`: 1 (legitimate) or -1 (phishing)

---

## 📡 API Documentation

### Endpoints

#### **GET /**
- **Description**: Root endpoint, redirects to Swagger documentation
- **Response**: `302 Redirect` → `/docs`

#### **POST /predict**
- **Description**: Batch prediction on uploaded CSV file
- **Request**:
  - **Type**: `multipart/form-data`
  - **Parameter**: `file` (CSV file)
- **Response**:
  - **Type**: `text/html`
  - **Body**: HTML table with predictions
  - **Side Effect**: Saves `prediction_output/output.csv`

**Example Response:**
```html
<table class="table table-striped">
  <thead>
    <tr><th>Feature1</th>...<th>predicted_column</th></tr>
  </thead>
  <tbody>
    <tr><td>-1</td>...<td>-1</td></tr>  <!-- Phishing -->
    <tr><td>1</td>...<td>1</td></tr>    <!-- Legitimate -->
  </tbody>
</table>
```

### CORS Configuration
- **Allowed Origins**: `*` (all origins)
- **Allowed Methods**: `*` (all HTTP methods)
- **Allowed Headers**: `*` (all headers)

---

## 📈 Model Performance

### Best Model Configuration

Based on 50 Optuna trials with 5-fold cross-validation:

**Selected Algorithm**: Random Forest / XGBoost (varies by run)

**Sample Performance Metrics**:

| Metric | Training Set | Test Set |
|--------|--------------|----------|
| **F1-Score** | 0.9845 | 0.9712 |
| **Precision** | 0.9821 | 0.9688 |
| **Recall** | 0.9869 | 0.9735 |
| **Accuracy** | 0.9850 | 0.9720 |

### Hyperparameter Optimization

The Optuna framework systematically searches the hyperparameter space:

```python
Search Space Example (Random Forest):
- n_estimators: [50, 300]
- max_depth: [5, 30]
- min_samples_split: [2, 20]
- min_samples_leaf: [1, 10]
- criterion: ['gini', 'entropy']

Optimization Process:
- Objective: Maximize F1-Score
- Cross-Validation: 5-fold stratified
- Number of Trials: 50
- Pruning: Enabled (MedianPruner)
```

### Model Artifacts

After training, the following artifacts are generated:

1. **model_evaluation_report.yaml**
```yaml
best_model: "random_forest" / "xgboost"
best_cv_f1_score: 0.9789
n_trials: 50
best_params:
  n_estimators: "200"
  max_depth: "15"
  min_samples_split: "5"
train_f1_score: 0.9845
train_precision: 0.9821
train_recall: 0.9869
test_f1_score: 0.9712
test_precision: 0.9688
test_recall: 0.9735
```

2. **Trained Model Files**
- `model.pkl`: Optimized classifier
- `preprocessor.pkl`: KNN imputer pipeline
- `NetworkModel`: Combined pipeline (preprocessor + model)

---

## 🌐 Deployment

### Docker Deployment

#### Build Docker Image
```bash
docker build -t phishing-detection:latest .
```

#### Run Container
```bash
docker run -d \
  -p 8080:8080 \
  --name phishing-api \
  -e MONGODB_URL_KEY="your_mongodb_connection_string" \
  phishing-detection:latest
```

#### Docker Compose (Optional)
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8080:8080"
    environment:
      - MONGODB_URL_KEY=${MONGODB_URL_KEY}
    restart: unless-stopped
```

### AWS Deployment Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  GitHub Repo    │─────▶│  GitHub Actions  │─────▶│   AWS ECR       │
│  (Code Push)    │      │  (CI/CD)         │      │  (Docker Image) │
└─────────────────┘      └──────────────────┘      └─────────────────┘
                                                             │
                                                             ▼
                         ┌──────────────────────────────────────────┐
                         │     AWS EC2 (Self-Hosted Runner)         │
                         │  • Pulls latest image from ECR           │
                         │  • Stops old container                   │
                         │  • Runs new container on port 8080       │
                         │  • Cleans up old images                  │
                         └──────────────────────────────────────────┘
```

### Deployment Steps

1. **AWS ECR Setup**
```bash
aws ecr create-repository --repository-name phishing-detection
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
```

2. **GitHub Secrets Configuration**

Add the following secrets to your GitHub repository:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `ECR_REPOSITORY_NAME`
- `MONGODB_URL_KEY`

3. **Self-Hosted Runner Setup** (on AWS EC2)
```bash
# Install Docker
sudo yum update -y
sudo yum install docker -y
sudo service docker start

# Configure GitHub Runner
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz
./config.sh --url https://github.com/sanjeevpatil804/End-to-End-intelligent-phishing-defense --token YOUR_TOKEN
./run.sh
```

---

## ⚙️ CI/CD Pipeline

### GitHub Actions Workflow

The `.github/workflows/main.yaml` defines a three-stage pipeline:

#### **Stage 1: Continuous Integration**
```yaml
- Checkout code from main branch
- Run linting checks
- Execute unit tests
- Validate code quality
```

#### **Stage 2: Continuous Delivery**
```yaml
- Install AWS CLI utilities
- Configure AWS credentials
- Login to Amazon ECR
- Build Docker image
- Tag image as 'latest'
- Push image to ECR repository
```

#### **Stage 3: Continuous Deployment**
```yaml
- Run on self-hosted EC2 runner
- Pull latest image from ECR
- Stop existing container (if running)
- Remove old container
- Run new container with environment variables
- Expose on port 8080
- Clean up unused images
```

### Trigger Conditions

The pipeline triggers on:
- **Push** to `main` branch
- **Excludes** changes to `README.md`

### Deployment Flow

```
Developer Push → GitHub → CI Tests → Build Docker → Push to ECR → 
Deploy to EC2 → Stop Old Container → Start New Container → Live
```

**Deployment Time**: ~5-8 minutes (from push to live)

---

## 🔮 Future Enhancements

### Model Improvements
- [ ] Implement deep learning models (LSTM, Transformer-based)
- [ ] Add real-time URL feature extraction API
- [ ] Ensemble methods with stacking/blending
- [ ] Online learning for continuous model updates
- [ ] A/B testing framework for model comparison

### Feature Engineering
- [ ] Extract additional features from DNS records
- [ ] Integrate VirusTotal API for reputation scores
- [ ] Add NLP features from page content analysis
- [ ] Implement WHOIS data integration
- [ ] Social media signals and brand verification

### Infrastructure & MLOps
- [ ] Implement model monitoring and drift detection
- [ ] Add Prometheus + Grafana for metrics
- [ ] Setup MLflow for experiment tracking
- [ ] Kubernetes deployment for auto-scaling
- [ ] Implement feature store (Feast/Tecton)
- [ ] Add model versioning and A/B deployment

### Application Features
- [ ] Real-time prediction API (single URL)
- [ ] Browser extension integration
- [ ] Email phishing detection module
- [ ] User feedback loop for model improvement
- [ ] Multi-language support for global URLs

### Data & Testing
- [ ] Expand dataset with recent phishing samples
- [ ] Implement data augmentation techniques
- [ ] Add comprehensive unit/integration tests
- [ ] Load testing and performance benchmarking
- [ ] Implement adversarial testing

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
```bash
git clone https://github.com/sanjeevpatil804/End-to-End-intelligent-phishing-defense.git
```

2. **Create a Feature Branch**
```bash
git checkout -b feature/your-feature-name
```

3. **Make Your Changes**
- Follow PEP 8 style guidelines
- Add docstrings to functions
- Update tests if applicable

4. **Commit Your Changes**
```bash
git add .
git commit -m "Add: Your descriptive commit message"
```

5. **Push to Your Fork**
```bash
git push origin feature/your-feature-name
```

6. **Create a Pull Request**
- Provide a clear description of changes
- Reference any related issues

### Code Quality Standards
- **Linting**: Ensure code passes flake8
- **Type Hints**: Use type annotations where possible
- **Documentation**: Update docstrings and README
- **Testing**: Add tests for new features

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Sanjeev Patil

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

## 👨‍💻 Author

**Sanjeev Patil**

- GitHub: [@sanjeevpatil804](https://github.com/sanjeevpatil804)
- LinkedIn: [Sanjeev Patil](https://www.linkedin.com/in/sanjeevpatil804)
- Email: sanjeevpatil804@aol.com

---

## 🙏 Acknowledgments

- **Dataset**: Phishing Website Dataset from UCI Machine Learning Repository
- **Frameworks**: FastAPI, scikit-learn, XGBoost, Optuna
- **Cloud Services**: MongoDB Atlas, AWS ECR/EC2
- **CI/CD**: GitHub Actions
- **Community**: Open-source contributors and cybersecurity researchers

---

## 📞 Support

If you encounter any issues or have questions:

1. **Check Documentation**: Review this README and code comments
2. **Search Issues**: Look for similar issues in the GitHub repository
3. **Open an Issue**: Create a detailed issue with:
   - Problem description
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, etc.)

---

## ⭐ Star the Repository

If you find this project useful, please consider giving it a star ⭐ on GitHub!

---

<div align="center">

**Built with ❤️ for a Safer Internet**

[🔝 Back to Top](#️-end-to-end-intelligent-phishing-defense)

</div>
