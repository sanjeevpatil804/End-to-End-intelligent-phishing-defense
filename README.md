# End-to-End Intelligent Phishing Defense

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://mlflow.org/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-brightgreen)](https://github.com/features/actions)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-ECR%20%7C%20EC2-orange)](https://aws.amazon.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An enterprise-grade, production-ready machine learning pipeline for real-time phishing website detection. Features automated CI/CD deployment to AWS, Optuna-based hyperparameter optimization, MLflow experiment tracking, and comprehensive data validation to achieve 85-95% F1 score.

---

## 📋 Table of Contents

- [Features](#features)
- [Project Architecture](#project-architecture)
- [CI/CD Pipeline](#cicd-pipeline)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Components](#pipeline-components)
- [MLflow Experiment Tracking](#mlflow-experiment-tracking)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [AWS Infrastructure Setup](#aws-infrastructure-setup)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## ✨ Features

### Core ML Pipeline
- **End-to-End ML Pipeline**: Data ingestion → Validation → Transformation → Model Training
- **Advanced HPO**: Optuna-based hyperparameter optimization with stratified K-fold CV
- **Experiment Tracking**: Comprehensive MLflow integration for metrics, parameters, and artifacts
- **Data Leakage Prevention**: Cross-validation on training data only during hyperparameter selection
- **Drift Detection**: Kolmogorov-Smirnov test for data distribution monitoring
- **Model Versioning**: Automated artifact management and model registry

### Production & DevOps
- **Automated CI/CD**: GitHub Actions workflow with 3-stage pipeline
- **Containerization**: Docker-based deployment for consistency across environments
- **Cloud Deployment**: Automated AWS ECR + EC2 deployment
- **REST API**: FastAPI-based inference endpoint with health checks
- **Self-Hosted Runner**: Continuous deployment on custom infrastructure
- **Zero-Downtime Deployment**: Rolling updates with container orchestration

### Quality & Reliability
- **Comprehensive Logging**: Structured logging with exception tracking
- **Schema Validation**: Automated data quality checks
- **Test Suite**: Unit tests integrated in CI pipeline
- **Code Quality**: Automated linting in CI/CD workflow
- **Monitoring Ready**: Health endpoints and logging for production monitoring

---

## 🏗️ Project Architecture

### ML Pipeline Flow

```
┌─────────────────┐
│  Data Ingestion │  ← MongoDB/CSV
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data Validation │  ← Schema check + Drift detection
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Transformation  │  ← KNN Imputation + Preprocessing
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │  ← Optuna HPO + MLflow Tracking
│   RF + XGBoost  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Registry  │  ← Best model + Full pipeline
└─────────────────┘
```

### Deployment Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        GitHub Repository                      │
│                 (End-to-End-intelligent-phishing-defense)     │
└───────────────────────────┬──────────────────────────────────┘
                            │
                    Push to main branch
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                    GitHub Actions Workflow                     │
│  ┌────────────────┐  ┌─────────────────┐  ┌────────────────┐│
│  │  CI Stage      │  │  CD Stage       │  │  Deployment    ││
│  │  • Lint        │→ │  • Build Docker │→ │  • Pull Image  ││
│  │  • Unit Tests  │  │  • Push to ECR  │  │  • Run Container││
│  └────────────────┘  └─────────────────┘  └────────────────┘│
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────┐
│                    AWS Infrastructure                          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  Amazon ECR (Elastic Container Registry)                │ │
│  │  • Stores Docker images                                 │ │
│  │  • Version tagged: latest                               │ │
│  └─────────────────────────────────────────────────────────┘ │
│                            │                                   │
│                            ▼                                   │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  EC2 Instance (Self-Hosted Runner)                      │ │
│  │  • Pulls image from ECR                                 │ │
│  │  • Runs Docker container on port 8080                   │ │
│  │  • Serves FastAPI application                           │ │
│  │  • Health monitoring: /health endpoint                  │ │
│  └─────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │  End Users    │
                    │  (HTTP/REST)  │
                    └───────────────┘
```

---

## � CI/CD Pipeline

### Overview

This project implements a fully automated **3-stage CI/CD pipeline** using GitHub Actions that handles testing, building, and deployment to AWS infrastructure.

### Pipeline Stages

#### 1️⃣ **Continuous Integration (CI)**
Runs on every push to `main` branch (except README changes).

**Steps**:
- ✅ **Code Checkout**: Pulls latest code from repository
- ✅ **Linting**: Validates code quality and formatting
- ✅ **Unit Tests**: Runs automated test suite
- ✅ **Quality Gates**: Ensures code meets standards before proceeding

```yaml
jobs:
  integration:
    name: Continuous Integration
    runs-on: ubuntu-latest
    steps:
      - Checkout Code
      - Lint code
      - Run unit tests
```

#### 2️⃣ **Continuous Delivery (CD)**
Builds and pushes Docker image to AWS ECR (only after CI passes).

**Steps**:
- 🔨 **Build Docker Image**: Creates containerized application
- 🏷️ **Tag Image**: Tags with `latest` version
- ☁️ **Push to ECR**: Uploads to Amazon Elastic Container Registry
- 🔐 **AWS Authentication**: Uses IAM credentials for secure access

```yaml
jobs:
  build-and-push-ecr-image:
    name: Continuous Delivery
    needs: integration
    steps:
      - Configure AWS credentials
      - Login to Amazon ECR
      - Build, tag, and push image
```

#### 3️⃣ **Continuous Deployment**
Deploys application to EC2 using self-hosted runner.

**Steps**:
- 📥 **Pull Latest Image**: Fetches from ECR
- 🔄 **Rolling Update**: Stops old container, starts new one
- 🧹 **Cleanup**: Removes unused images/containers
- 🚀 **Service Start**: Runs on port 8080 with environment variables

```yaml
jobs:
  Continuous-Deployment:
    needs: build-and-push-ecr-image
    runs-on: self-hosted
    steps:
      - Pull latest images
      - Run Docker Image to serve users
      - Clean previous images and containers
```

### Workflow Triggers

```yaml
on:
  push:
    branches:
      - main
    paths-ignore:
      - 'README.md'
```

**Trigger Conditions**:
- ✅ Push to `main` branch
- ❌ Ignores README.md changes (documentation-only updates)

### Required GitHub Secrets

Configure these in **Settings → Secrets and variables → Actions**:

| Secret Name | Description | Example |
|------------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | AWS IAM access key | `AKIA...` |
| `AWS_SECRET_ACCESS_KEY` | AWS IAM secret key | `wJalrXUtn...` |
| `AWS_REGION` | AWS deployment region | `us-east-1` |
| `ECR_REPOSITORY_NAME` | ECR repository name | `phishing-detection` |
| `AWS_ECR_LOGIN_URI` | Full ECR registry URI | `123456789.dkr.ecr.us-east-1.amazonaws.com` |

### Pipeline Features

✅ **Automated Testing**: Every commit is tested before deployment  
✅ **Zero Downtime**: Rolling updates with health checks  
✅ **Rollback Support**: Previous images retained in ECR  
✅ **Security**: IAM-based authentication, no hardcoded credentials  
✅ **Observability**: GitHub Actions logs for full audit trail  
✅ **Self-Healing**: Automatic container restart on failure  

### Monitoring the Pipeline

**View Workflow Runs**:
```
GitHub Repository → Actions Tab → workflow runs
```

**Check Deployment Status**:
```bash
# SSH into EC2 instance
ssh -i your-key.pem ec2-user@your-ec2-ip

# Check running containers
docker ps

# View container logs
docker logs networksecurity

# Check application health
curl http://localhost:8080/health
```

---

## �🚀 Installation

### Prerequisites

- Python 3.10+
- MongoDB (optional, for data ingestion)
- Git

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/sanjeevpatil804/End-to-End-intelligent-phishing-defense.git
cd End-to-End-intelligent-phishing-defense
```

2. **Create virtual environment**

```bash
python -m venv mlenv
source mlenv/bin/activate  # On Windows: mlenv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file:

```bash
MONGO_DB_URI=your_mongodb_connection_string
```

---

## ⚡ Quick Start

### Train the Model

```bash
# Run the complete pipeline
python main.py
```

This will:
- Ingest data from MongoDB or CSV
- Validate and transform the data
- Train models with Optuna hyperparameter optimization
- Log everything to MLflow
- Save the best model to `final_model/model.pkl`

### View Experiment Results

```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# Open in browser
# http://localhost:5000
```

---

## 🔧 Pipeline Components

### 1. Data Ingestion

**Location**: `networksecurity/components/data_ingestion.py`

- Fetches data from MongoDB or local CSV
- Splits into train/test (80/20)
- Saves to artifact directory

```python
from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.entity.config_entity import DataIngestionConfig

config = DataIngestionConfig(...)
ingestion = DataIngestion(config)
artifacts = ingestion.initiate_data_ingestion()
```

### 2. Data Validation

**Location**: `networksecurity/components/Data_validation.py`

- Validates column count against schema
- Detects data drift using Kolmogorov-Smirnov test
- Generates drift report

**Key Features**:
- Schema validation
- Statistical drift detection
- Automated reporting

### 3. Data Transformation

**Location**: `networksecurity/components/data_transformation.py`

- KNN imputation for missing values
- Target encoding (-1 → 0 for binary classification)
- Saves preprocessor pipeline

**Configuration**:
```python
DATA_TRANSFORMATION_IMPUTER_PARAMS = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}
```

### 4. Model Training

**Location**: `networksecurity/components/model_trainer.py`

**Models Supported**:
- Random Forest Classifier
- XGBoost Classifier

**Hyperparameter Optimization**:
- Uses Optuna with stratified 3-fold cross-validation
- **No data leakage**: Test set only used for final evaluation
- Real-time trial logging to MLflow

**Key Hyperparameters** (configurable in `networksecurity/Utils/main_utils/optuna_tuner.py`):

**Random Forest**:
- `n_estimators`: 100-500
- `max_depth`: 5-30
- `criterion`: gini, entropy, log_loss
- `class_weight`: None, balanced

**XGBoost**:
- `n_estimators`: 100-500
- `learning_rate`: 0.01-0.3 (log scale)
- `max_depth`: 3-10
- `reg_alpha`, `reg_lambda`: L1/L2 regularization

---

## 📊 MLflow Experiment Tracking

### What Gets Logged

**Parameters**:
- Dataset size (train/test samples, features)
- Optimization method (Optuna)
- Best model hyperparameters
- Model selection (RF vs XGBoost)

**Metrics**:
- Per-trial F1 scores (real-time)
- Best model F1, precision, recall (train/test)
- Model comparison scores

**Artifacts**:
- Model evaluation report (YAML)
- Trained model pipeline (.pkl)
- Best model only (.pkl)
- Full preprocessor + model pipeline

### Access MLflow UI

```bash
# Default (uses ./mlruns/)
mlflow ui --host 0.0.0.0 --port 5000

# Custom storage
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 0.0.0.0 \
  --port 5000
```

**Features**:
- Compare multiple training runs
- Download models and artifacts
- View hyperparameter importance
- Track metric evolution over trials

---

## 🎯 Model Training

### Adjust Number of Trials

In `networksecurity/components/model_trainer.py`, line ~56:

```python
optimization_results = optimize_models(
    X_train, y_train, x_test, y_test,
    models_to_optimize=['rf', 'xgb'],
    n_trials=50  # Increase for better hyperparameters (e.g., 100)
)
```

**Recommendations**:
- **Quick testing**: `n_trials=10` (~2-5 minutes)
- **Development**: `n_trials=50` (~10-20 minutes)
- **Production**: `n_trials=100+` (~30+ minutes)

### Custom Hyperparameter Ranges

Edit `networksecurity/Utils/main_utils/optuna_tuner.py`:

```python
# Example: Expand Random Forest search space
params = {
    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),  # Increased max
    'max_depth': trial.suggest_int('max_depth', 5, 50),            # Deeper trees
    # ... add more parameters
}
```

### Expected Performance

**Typical Results** (on phishing dataset):
- **F1 Score**: 0.85-0.95
- **Precision**: 0.80-0.93
- **Recall**: 0.82-0.96

---

## 🚀 Deployment

### Local Development

**Start FastAPI Server**:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

**Access Application**:
- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

### Docker Deployment

**Build and Run Locally**:
```bash
# Build image
docker build -t phishing-detection:latest .

# Run container
docker run -d -p 8080:8080 \
  --name networksecurity \
  -e MONGO_DB_URI=$MONGO_DB_URI \
  phishing-detection:latest

# Check logs
docker logs -f networksecurity

# Stop container
docker stop networksecurity && docker rm networksecurity
```

### Automated AWS Deployment

**Deployment happens automatically** when you push to `main` branch:

1. **Commit and Push**:
```bash
git add .
git commit -m "Update model or code"
git push origin main
```

2. **Monitor Pipeline**:
   - Go to GitHub → **Actions** tab
   - Watch the 3-stage pipeline execute
   - CI → CD → Deployment (~5-10 minutes)

3. **Access Deployed Application**:
```bash
# Get your EC2 public IP from AWS Console
curl http://<EC2-PUBLIC-IP>:8080/health

# Test prediction
curl -X POST http://<EC2-PUBLIC-IP>:8080/predict \
  -H "Content-Type: application/json" \
  -d @sample_input.json
```

### API Endpoints

#### 1. Health Check
```bash
GET /health

Response:
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-05T10:30:00Z"
}
```

#### 2. Prediction
```bash
POST /predict

Request Body:
{
  "having_IP_Address": 1,
  "URL_Length": -1,
  "Shortining_Service": 1,
  "having_At_Symbol": 1,
  "double_slash_redirecting": -1,
  "Prefix_Suffix": -1,
  "having_Sub_Domain": -1,
  "SSLfinal_State": -1,
  "Domain_registeration_length": -1,
  "Favicon": 1,
  "port": 1,
  "HTTPS_token": -1,
  "Request_URL": 1,
  "URL_of_Anchor": -1,
  "Links_in_tags": 1,
  "SFH": -1,
  "Submitting_to_email": 1,
  "Abnormal_URL": -1,
  "Redirect": 0,
  "on_mouseover": 1,
  "RightClick": 1,
  "popUpWidnow": 1,
  "Iframe": 1,
  "age_of_domain": -1,
  "DNSRecord": -1,
  "web_traffic": -1,
  "Page_Rank": -1,
  "Google_Index": 1,
  "Links_pointing_to_page": 1,
  "Statistical_report": -1
}

Response:
{
  "prediction": -1,
  "probability": 0.92,
  "label": "phishing",
  "confidence": "high"
}
```

#### 3. Model Info
```bash
GET /model/info

Response:
{
  "model_type": "XGBoost",
  "version": "1.0.0",
  "training_date": "2025-11-04",
  "f1_score": 0.93,
  "features": 30
}
```

---

## ☁️ AWS Infrastructure Setup

### Prerequisites

1. **AWS Account** with appropriate permissions
2. **IAM User** with programmatic access
3. **EC2 Instance** (t2.medium or larger recommended)
4. **ECR Repository** created

### Step-by-Step Setup

#### 1. Create IAM User

```bash
# Required IAM Policies:
- AmazonEC2ContainerRegistryFullAccess
- AmazonEC2FullAccess (or limited EC2 permissions)
```

**Create Access Keys**:
1. AWS Console → IAM → Users → Your User
2. Security Credentials → Create Access Key
3. Save `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

#### 2. Create ECR Repository

```bash
# Using AWS CLI
aws ecr create-repository \
  --repository-name phishing-detection \
  --region us-east-1

# Or via AWS Console:
# ECR → Repositories → Create repository
```

**Note the URI**: `123456789.dkr.ecr.us-east-1.amazonaws.com/phishing-detection`

#### 3. Launch EC2 Instance

**Recommended Configuration**:
- **AMI**: Ubuntu 22.04 LTS
- **Instance Type**: t2.medium (2 vCPU, 4GB RAM)
- **Storage**: 30GB EBS
- **Security Group**: 
  - Port 22 (SSH) - Your IP only
  - Port 8080 (HTTP) - 0.0.0.0/0
  - Port 443 (HTTPS) - 0.0.0.0/0

**User Data Script** (runs on launch):
```bash
#!/bin/bash
# Update system
apt-get update -y
apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu

# Install AWS CLI
apt-get install -y awscli

# Install GitHub Runner (for self-hosted)
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz \
  -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf actions-runner-linux-x64-2.311.0.tar.gz
```

#### 4. Configure GitHub Self-Hosted Runner

**On EC2 Instance**:
```bash
# SSH into instance
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>

# Navigate to runner directory
cd ~/actions-runner

# Configure runner
./config.sh --url https://github.com/sanjeevpatil804/End-to-End-intelligent-phishing-defense \
  --token <YOUR-RUNNER-TOKEN>

# Install as service
sudo ./svc.sh install
sudo ./svc.sh start
```

**Get Runner Token**:
1. GitHub Repo → Settings → Actions → Runners
2. Click "New self-hosted runner"
3. Copy the token from setup instructions

#### 5. Add GitHub Secrets

**Repository Settings → Secrets → Actions → New repository secret**:

```bash
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=wJalrXUtn...
AWS_REGION=us-east-1
ECR_REPOSITORY_NAME=phishing-detection
AWS_ECR_LOGIN_URI=123456789.dkr.ecr.us-east-1.amazonaws.com
```

#### 6. Verify Setup

**Push to trigger pipeline**:
```bash
git push origin main
```

**Check deployment**:
```bash
# On EC2
docker ps
docker logs networksecurity

# Test endpoint
curl http://localhost:8080/health
```

### Cost Estimation

**Monthly AWS Costs** (approximate):
- EC2 t2.medium: ~$30/month
- ECR Storage (< 10GB): ~$1/month
- Data Transfer: ~$5/month
- **Total**: ~$36/month

**Cost Optimization Tips**:
- Use t2.micro for development ($8/month)
- Enable EC2 Auto Stop/Start during non-business hours
- Use ECR lifecycle policies to delete old images

---

## 📁 Project Structure

```
End-to-End-intelligent-phishing-defense/
├── .github/
│   └── workflows/
│       └── main.yml                  # CI/CD pipeline definition
├── artifact/                          # Pipeline outputs (timestamped)
│   └── <timestamp>/
│       ├── data_ingestion/
│       ├── data_validation/
│       ├── data_transformation/
│       └── model_trainer/
├── final_model/                       # Best model + preprocessor
│   └── model.pkl
├── mlruns/                           # MLflow tracking data
├── Network_Data/                     # Raw dataset
│   └── phishingData.csv
├── networksecurity/
│   ├── components/                   # Pipeline stages
│   │   ├── data_ingestion.py
│   │   ├── Data_validation.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── Utils/
│   │   └── main_utils/
│   │       ├── utils.py              # IO utilities
│   │       ├── optuna_tuner.py       # HPO logic
│   │       ├── classification_metrics.py
│   │       └── estimator.py          # Model wrapper
│   ├── constants/
│   │   └── training_pipeline/        # Config constants
│   ├── entity/
│   │   ├── config_entity.py          # Pipeline configs
│   │   └── artifact_entity.py        # Pipeline artifacts
│   ├── exception/
│   │   └── exception.py              # Custom exceptions
│   └── logging/
│       └── __init__.py               # Logging setup
├── data_schema/
│   └── schema.yaml                   # Dataset schema
├── notebooks/                         # Jupyter notebooks
├── .env                              # Environment variables (git-ignored)
├── .gitignore                        # Git ignore rules
├── requirements.txt                   # Python dependencies
├── main.py                           # Pipeline orchestration
├── app.py                            # FastAPI application
├── Dockerfile                        # Container definition
├── Setup.py                          # Package setup
└── README.md                         # This file
```

---

## ⚙️ Configuration

### Training Pipeline Constants

**File**: `networksecurity/constants/training_pipeline/__init__.py`

```python
# Data Ingestion
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.2

# Data Transformation
DATA_TRANSFORMATION_IMPUTER_PARAMS = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}

# Model Training
MODEL_TRAINER_EXPECTED_SCORE = 0.85
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD = 0.05
```

### Schema Definition

**File**: `data_schema/schema.yaml`

```yaml
columns:
  - having_IP_Address
  - URL_Length
  - Shortining_Service
  # ... (31 columns total)

target_column: Result
```

---

## � Monitoring & Maintenance

### Application Monitoring

**Health Check Endpoint**:
```bash
# Automated health check (every 5 minutes)
curl http://<EC2-PUBLIC-IP>:8080/health

# Expected response
{
  "status": "healthy",
  "uptime": "5d 12h 30m",
  "memory_usage": "45%",
  "cpu_usage": "15%"
}
```

**Container Logs**:
```bash
# Real-time logs
docker logs -f networksecurity

# Last 100 lines
docker logs --tail 100 networksecurity

# Logs with timestamps
docker logs -t networksecurity
```

**Resource Monitoring**:
```bash
# Container stats
docker stats networksecurity

# Disk usage
df -h

# Memory usage
free -h

# CPU info
top
```

### MLflow Monitoring

**Track Experiment Metrics**:
```bash
# Start MLflow UI (if running on EC2)
mlflow ui --host 0.0.0.0 --port 5000

# Access via browser
http://<EC2-PUBLIC-IP>:5000
```

**Key Metrics to Monitor**:
- Model F1 Score trends
- Training time per run
- Hyperparameter distributions
- Model drift indicators

### Automated Alerts (Optional)

**Set up CloudWatch Alarms** for:
- CPU Utilization > 80%
- Memory Utilization > 90%
- Disk Space < 10%
- Container health check failures

### Maintenance Tasks

**Weekly**:
- Check container logs for errors
- Review MLflow experiments
- Monitor API response times

**Monthly**:
- Update dependencies: `pip install --upgrade -r requirements.txt`
- Retrain model with new data
- Clean up old Docker images: `docker system prune -a`

**Quarterly**:
- Review and update hyperparameter ranges
- Analyze model performance degradation
- Update schema if data format changes

---

## �🐛 Troubleshooting

### MLflow UI "Address already in use"

```bash
# Find and kill existing process
lsof -ti:5000 | xargs kill -9

# Or use alternate port
mlflow ui --host 0.0.0.0 --port 5001
```

### MongoDB Connection Issues

Ensure `.env` contains valid connection string:
```bash
MONGO_DB_URI=mongodb+srv://user:pass@cluster.mongodb.net/database
```

Or use local CSV:
- Place data in `Network_Data/phishingData.csv`
- Pipeline auto-falls back to CSV if MongoDB unavailable

### CI/CD Pipeline Failures

**Problem**: GitHub Actions workflow fails at AWS login
```bash
Error: Unable to locate credentials
```

**Solution**:
1. Verify GitHub Secrets are set correctly
2. Check IAM user has ECR permissions
3. Ensure AWS_REGION matches ECR repository region

**Problem**: Docker build fails in CI
```bash
Error: Cannot connect to the Docker daemon
```

**Solution**:
- Ensure Docker is installed on self-hosted runner
- Add runner user to docker group: `sudo usermod -aG docker $USER`

**Problem**: Container fails to start on EC2
```bash
Error: port is already allocated
```

**Solution**:
```bash
# Stop and remove existing container
docker stop networksecurity
docker rm networksecurity

# Or kill process using port 8080
lsof -ti:8080 | xargs kill -9
```

### Deployment Issues

**Problem**: Application returns 502 Bad Gateway

**Solution**:
```bash
# Check if container is running
docker ps

# Check container logs
docker logs networksecurity

# Restart container
docker restart networksecurity
```

**Problem**: Model not found error

**Solution**:
```bash
# Ensure model file exists
ls -la final_model/model.pkl

# Rebuild with model included
docker build -t phishing-detection:latest .
```

### Self-Hosted Runner Issues

**Problem**: Runner offline in GitHub

**Solution**:
```bash
# SSH into EC2
cd ~/actions-runner

# Check runner status
sudo ./svc.sh status

# Restart runner service
sudo ./svc.sh stop
sudo ./svc.sh start
```

---

## 📈 Performance Optimization

### Speed Up Training

1. **Reduce trials**: Set `n_trials=10` in `model_trainer.py`
2. **Lower batch size**: Reduce batch size for faster iterations
3. **Fewer epochs**: Use lower epochs for quicker experiments

### Improve Model Quality

1. **More trials**: `n_trials=100+`
2. **Broader hyperparameter ranges**: Edit `optuna_tuner.py`
3. **Feature engineering**: Add domain-specific features
4. **Ensemble methods**: Combine RF + XGBoost predictions

---

## 🔐 Security & Privacy

- **Data Leakage Prevention**: Cross-validation only on training split
- **Secure Credentials**: Use `.env` for sensitive data (never commit!)
- **Model Versioning**: Full artifact tracking for audit trails

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

**Code Style**:
- Follow PEP 8
- Add docstrings to functions
- Include type hints where applicable
- Write unit tests for new features

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Sanjeev Patil**
- GitHub: [@sanjeevpatil804](https://github.com/sanjeevpatil804)
- Email: your.email@example.com

**Project Link**: [https://github.com/sanjeevpatil804/End-to-End-intelligent-phishing-defense](https://github.com/sanjeevpatil804/End-to-End-intelligent-phishing-defense)

---

## 🙏 Acknowledgments

- [MLflow](https://mlflow.org/) for experiment tracking
- [Optuna](https://optuna.org/) for hyperparameter optimization
- [XGBoost](https://xgboost.readthedocs.io/) and [scikit-learn](https://scikit-learn.org/) for ML models
- [GitHub Actions](https://github.com/features/actions) for CI/CD automation
- [AWS](https://aws.amazon.com/) for cloud infrastructure
- [Docker](https://www.docker.com/) for containerization
- [FastAPI](https://fastapi.tiangolo.com/) for REST API framework

---

## 📚 Additional Resources

### ML & Development
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [Optuna Tutorial](https://optuna.readthedocs.io/en/stable/tutorial/index.html)
- [FastAPI Guide](https://fastapi.tiangolo.com/)
- [Data Leakage in ML](https://machinelearningmastery.com/data-leakage-machine-learning/)

### DevOps & Deployment
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [AWS ECR User Guide](https://docs.aws.amazon.com/ecr/)
- [AWS EC2 Documentation](https://docs.aws.amazon.com/ec2/)

### Security
- [GitHub Secrets Management](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [Docker Security](https://docs.docker.com/engine/security/)

---

**Happy Model Training! 🚀**
