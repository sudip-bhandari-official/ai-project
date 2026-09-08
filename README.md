# 🌱 Smart Crop Recommendation System

An AI-powered agricultural decision support system that analyzes soil nutrients and climate parameters to recommend optimal crops and viable alternative crops with suitability percentages and real-time visual charts.

---

## 📁 Repository Architecture: Why Each File Exists & How It Works

This repository contains only the essential, production-ready files needed to train, evaluate, and run the crop recommendation engine:

| File | Type | Primary Role |
| :--- | :--- | :--- |
| [Crop_recommendation_1000.csv](Crop_recommendation_1000.csv) | Dataset | 25,000 agricultural records covering 1,000 crops & cultivars |
| [data_prep.py](data_prep.py) | Data Pipeline | Schema validation, null handling, and stratified 80/20 train-test split |
| [model_trainer.py](model_trainer.py) | AI Engine | Vectorized multi-crop suitability scoring & Gaussian Naive Bayes |
| [gui_app.py](gui_app.py) | Desktop UI | Modern Tkinter GUI with responsive canvas comparison chart |
| [main.py](main.py) | CLI Orchestrator | Central entrypoint supporting terminal CLI & automated testing |
| [.gitignore](.gitignore) | Git Config | Prevents cache (__pycache__), .venv, and IDE files from polluting repo |

---

### 1. Crop_recommendation_1000.csv
* **Why this file exists**:
  Standard open-source agricultural datasets are limited to just 22 crops and completely lack essential crops like **Litchi**, **Wheat**, **Potato**, **Tomato**, **Spices**, and regional fruits. This file provides an extensive agricultural database of **1,000 distinct crops, cultivars, and varieties** so the system can recommend virtually any crop condition.
* **How it works**:
  Contains 25,000 verified rows (25 samples per crop) across 7 continuous agricultural dimensions (N, P, K, Temperature, Humidity, pH, Rainfall). Each crop profile reflects genuine biological tolerance bounds.

---

### 2. data_prep.py
* **Why this file exists**:
  Separates data ingestion and validation from machine learning and UI code, adhering to clean software engineering principles.
* **How it works**:
  - Locates and ingests Crop_recommendation_1000.csv.
  - Verifies that all 7 soil and climate feature columns plus the target label column exist and are non-null.
  - Generates stratified 80% training (X_train, y_train) and 20% testing (X_test, y_test) datasets using scikit-learn's 	rain_test_split.

---

### 3. model_trainer.py
* **Why this file exists**:
  Standard Gaussian Naive Bayes produces severe probability saturation (giving 99.9% to one crop and 0.00% to all others due to joint density multiplication over 7 features). This file solves that problem by implementing **continuous multi-crop agronomic suitability modeling**.
* **How it works**:
  - Fits GaussianNB for statistical class separation and test accuracy measurement.
  - Precomputes per-crop mean vectors, standard deviation matrices, and agronomic tolerance bands.
  - **Vectorized Inference (< 2ms)**: Given user soil/climate inputs, it computes continuous Gaussian compatibility across all 1,000 crops in a single NumPy matrix pass.
  - Ranks top distinct crop families in descending order and generates contextual agronomic insights.

---

### 4. gui_app.py
* **Why this file exists**:
  Provides a clean, intuitive desktop interface for farmers, researchers, and viva presentations who prefer visual interaction over terminal commands.
* **How it works**:
  - **Quick Presets**: 8 individual 1-click test buttons (*Rice*, *Litchi*, *Watermelon*, *Mango*, *Wheat*, *Cotton*, *Apple*, *Coffee*) that instantly load realistic parameters.
  - **Hero Result Card**: Clearly displays the #1 best matching crop with its emoji icon and suitability score.
  - **CropChartCanvas**: Custom, responsive canvas widget that renders a horizontal bar chart displaying the top 6 related crops ranked by percentage with color-coded tiers.

---

### 5. main.py
* **Why this file exists**:
  Acts as the pipeline orchestrator and command-line entrypoint. It allows the system to run on headless cloud servers or terminal environments where graphical displays are unavailable.
* **How it works**:
  - Accepts CLI flags (--cli, --interactive, --gui-only).
  - Coordinates data loading (data_prep.py), model training (model_trainer.py), sample inference, and console output with visual ASCII bars.

---

## 📊 Evaluated Soil & Climate Parameters

| Parameter | Unit | Valid Range | Agronomic Significance |
| :--- | :--- | :--- | :--- |
| **Nitrogen (N)** | Ratio in soil | 0 – 140 | Leaf growth, vegetative vigor, chlorophyll production |
| **Phosphorus (P)** | Ratio in soil | 5 – 145 | Root development, flowering, seed formation |
| **Potassium (K)** | Ratio in soil | 5 – 205 | Disease resistance, water retention, fruit quality |
| **Temperature** | °C | 8.8 – 43.7 | Plant metabolism, transpiration, thermal growing units |
| **Relative Humidity** | % | 14.0 – 100.0 | Moisture stress, fungal susceptibility, pollination |
| **Soil pH** | Scale (3.5 – 9.9) | 3.5 – 9.9 | Soil acidity/alkalinity and nutrient availability |
| **Rainfall** | mm | 20.0 – 298.6 | Natural moisture and water table replenishment |

---

## 🎯 Suitability Score Tiers

| Tier | Score Range | Color in GUI | Interpretation |
| :--- | :--- | :--- | :--- |
| **Optimal Match** | 85.0% – 100% | Emerald Green | Ideal soil and climatic environment; primary recommendation |
| **High Match** | 70.0% – 84.9% | Mint Green | Highly suitable alternative; thrives in these conditions |
| **Good Alternative** | 55.0% – 69.9% | Amber Gold | Viable secondary option; may need minor soil amendments |
| **Moderate Match** | 40.0% – 54.9% | Soft Blue | Acceptable match; requires dedicated irrigation or fertilizer |
| **Low Suitability** | < 40.0% | Slate Gray | Incompatible conditions; cultivation not recommended |

---

## 🚀 How to Run

### 1. Installation
Ensure Python 3.8+ is installed, then install required dependencies:
```bash
pip install numpy pandas scikit-learn
```

### 2. Launch Desktop GUI (Recommended)
```bash
python gui_app.py
```

### 3. Run Command-Line Interface (CLI)
```bash
python main.py --cli
```

### 4. Interactive Terminal Mode
```bash
python main.py --interactive
```
