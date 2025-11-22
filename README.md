🚀 UNSW-NB15 Intrusion Detection System (IDS)

This project implements a complete Machine Learning–based Intrusion Detection System (IDS) using the UNSW-NB15 cybersecurity dataset, one of the most widely used datasets in network security research.

The goal is to train, evaluate, and compare ML models capable of detecting network intrusions with high accuracy — and prepare the groundwork for real-time intrusion detection.

📌 Project Objectives

Clean and preprocess the full UNSW-NB15 dataset (~2.5M rows)

Encode categorical network features

Train two ML models:

Random Forest Classifier

XGBoost Classifier

Compare:

Accuracy

Precision, Recall, F1-score

Generate:

Confusion matrices

Feature importance plot

Save trained models for potential real-time predictions

📂 Project Structure
UNSW-NB15-Intrusion-Detection-System/
│
├── src/                     # Scripts (training, preprocessing)
├── data/                    # (ignored) dataset placeholder
├── models/                  # (ignored) trained models
├── images/                  # confusion matrices, plots
├── results/                 # evaluation outputs
│
├── ids_model.py             # Main pipeline (RF + XGBoost)
├── predict_packet.py        # Load model & predict flows
├── feature_importance.png   # Top features plot
├── rf_confusion_matrix.png
├── xgb_confusion_matrix.png
│
└── .gitignore


⚠️ Large dataset and model files are excluded using .gitignore.

🧹 Data Preprocessing

This project includes full preprocessing:

Remove rows with missing labels

Encode categorical columns (proto, state, service, attack_cat)

Convert numerical fields safely (errors='coerce')

Merge 4 UNSW-NB15 parts

Remove IP address fields

Apply stratified train/test split

🤖 Models Used
1️⃣ Random Forest

200 estimators

Parallel training

Strong performance on tabular data

2️⃣ XGBoost

Optimized gradient boosting

Excellent generalization

Highly effective on large datasets

Both models achieve > 99.9% accuracy with balanced precision/recall.

📊 Evaluation Metrics

The pipeline generates:

✔️ Accuracy

✔️ Precision / Recall / F1-score

✔️ Confusion matrix (PNG)

✔️ Feature importance ranking

Example files:

rf_confusion_matrix.png

xgb_confusion_matrix.png

📦 Trained Models

Models saved as:

models/
├── rf_ids_model.pkl
└── xgb_ids_model.pkl


Both can be loaded via predict_packet.py.

🔮 Next Steps (Planned Enhancements)

This repository will be extended with:

Real-time packet sniffing (Scapy)

Deep learning model (LSTM or 1D-CNN)

REST API for predictions (FastAPI/Flask)

Docker container for deployment

Streamlit interactive dashboard

🧑‍💻 Running the Project

Download UNSW-NB15 dataset from the official source

Place CSV files under data/

Train models:

python ids_model.py


Predict a new network flow:

python predict_packet.py

📫 Contact

El Mehdi El Afghani
📧 elafghani1111@gmail.com

🔗 GitHub: https://github.com/croki2
