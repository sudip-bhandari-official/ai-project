# ?? Smart Crop Recommendation System

An AI-powered agricultural decision support system that analyzes soil nutrients and climate parameters to recommend optimal crops and viable alternatives with suitability percentages and real-time visual charts.

---

## ?? Key Features

- **1,000 Agricultural Crops & Varieties**: Extensively cataloged across Cereals, Pulses, Fruits, Vegetables, Spices, and Cash crops (including Litchi, Wheat, Potato, Mango, Watermelon, Rice, and regional cultivars).
- **Multi-Crop Compatibility Engine**: Replaces single-crop saturation with continuous Gaussian suitability modeling. Crops sharing similar climate tolerances (e.g. Watermelon & Muskmelon, or Rice & Jute) display their respective match percentages.
- **Interactive Visual Comparison Chart**: Built-in responsive horizontal bar graph in the desktop GUI ranking the top matching crops in descending order with color-coded suitability tiers:
  - **Optimal Match (=85%)**
  - **High Match (70–84%)**
  - **Good Alternative (55–69%)**
  - **Moderate Match (40–54%)**
- **Sub-2ms Vectorized Inference**: High-speed NumPy matrix operations enable instant prediction across all 1,000 crops.
- **Quick Individual Presets**: 1-click presets for rapid testing (*Rice*, *Litchi*, *Watermelon*, *Mango*, *Wheat*, *Cotton*, *Apple*, *Coffee*).
- **Dual Interface**: Interactive Tkinter Desktop GUI and Terminal CLI mode.

---

## ?? Environmental & Soil Parameters

The system evaluates 7 critical agricultural factors:
1. **Nitrogen (N)**: Soil ratio (0 – 140)
2. **Phosphorus (P)**: Soil ratio (5 – 145)
3. **Potassium (K)**: Soil ratio (5 – 205)
4. **Temperature**: Air temperature in °C (8.8 – 43.7)
5. **Relative Humidity**: Air humidity in % (14.0 – 100.0)
6. **Soil pH**: Acidity/alkalinity scale (3.5 – 9.9)
7. **Rainfall**: Precipitation in mm (20.0 – 298.6)

---

## ?? Getting Started

### 1. Prerequisites
Ensure Python 3.8+ is installed with the required libraries:
`ash
pip install numpy pandas scikit-learn
`

### 2. Launching Desktop GUI
To open the graphical interface with the visual bar chart:
`ash
python gui_app.py
`

### 3. Running Terminal CLI Mode
To run predictions directly from the command line:
`ash
python main.py --cli
`
Or for interactive step-by-step inputs:
`ash
python main.py --interactive
`

---

## ?? Project Structure

- gui_app.py: Desktop GUI with responsive canvas comparison chart and preset selectors.
- model_trainer.py: Gaussian Naive Bayes classifier, vectorized suitability engine, and inference helpers.
- data_prep.py: Dataset loading, schema validation, and train/test split.
- main.py: Pipeline orchestrator and CLI entrypoint.
- crop_data_1000.py: Dataset generator for 1,000 agricultural crops.
- Crop_recommendation_1000.csv: Expanded 1,000 crops dataset (25,000 verified samples).
- Crop_recommendation.csv: Standard 22 crops baseline dataset (2,200 samples).
