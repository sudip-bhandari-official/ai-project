# 🌱 Smart Crop Recommendation System — Complete Project Reference (Viva Prep)

A one-stop cheat sheet of **everything used in this project**: the tech stack, every library,
every file, every function, where it lives, and how the whole thing works.

---

## 1. What the project does (one-line answer)

> "It is an AI-based crop recommendation system. The user enters 7 soil and climate values
> (N, P, K, temperature, humidity, pH, rainfall) and the system recommends the best crop plus
> alternative crops, each with a suitability percentage, shown in a desktop GUI and a terminal CLI."

---

## 2. Tech Stack (what to say when asked "what did you use?")

| Layer | Technology Used |
| :--- | :--- |
| **Programming Language** | Python 3 (tested on Python 3.11) |
| **Machine Learning** | scikit-learn (Gaussian Naive Bayes) |
| **Numerical Computing** | NumPy (vectorized scoring math) |
| **Data Handling** | pandas (reading CSV, DataFrames) |
| **Desktop GUI** | Tkinter (built into Python) + ttk |
| **CLI / Arguments** | argparse, sys (Python standard library) |
| **File Handling** | os (Python standard library) |
| **Dataset Format** | CSV (Comma Separated Values) |
| **ML Algorithm** | Gaussian Naive Bayes + custom Gaussian suitability scoring |

**Install command:** `pip install numpy pandas scikit-learn`
(Tkinter, os, sys, argparse come built-in with Python — nothing to install.)

---

## 3. Libraries used and WHY (with exact location)

| Library | Imported in | Why it is used |
| :--- | :--- | :--- |
| `os` | data_prep.py | Check if the dataset file exists, build file paths |
| `pandas` (`pd`) | data_prep.py, model_trainer.py | Read CSV, store data in DataFrames, group by crop |
| `sklearn.model_selection.train_test_split` | data_prep.py | Split data into 80% train / 20% test (stratified) |
| `numpy` (`np`) | model_trainer.py | Fast vectorized math to score all 1000 crops at once |
| `sklearn.naive_bayes.GaussianNB` | model_trainer.py | The machine learning classifier model |
| `sklearn.metrics.accuracy_score` | model_trainer.py | Measure test accuracy |
| `sklearn.metrics.classification_report` | model_trainer.py | Precision, recall, F1-score report |
| `tkinter` (`tk`) | gui_app.py | Build the desktop window and widgets |
| `tkinter.ttk`, `tkinter.messagebox` | gui_app.py | Styled widgets + popup warning/error boxes |
| `sys` | main.py | Fix console encoding (for emojis/Unicode) |
| `argparse` | main.py | Read command-line flags like --cli, --interactive |

---

## 4. Files in the project (what each one is for)

| File | Role | Lines |
| :--- | :--- | :--- |
| `Crop_recommendation_1000.csv` | **Dataset** — 25,000 rows, 1,000 crops, 7 features + label | ~25,001 |
| `data_prep.py` | **Data pipeline** — load, validate, clean, split | ~58 |
| `model_trainer.py` | **AI engine** — train model + suitability scoring | ~259 |
| `gui_app.py` | **Desktop GUI** — Tkinter interface + bar chart | ~571 |
| `main.py` | **Orchestrator / CLI** — ties everything, terminal mode | ~131 |
| `.gitignore` | Git config — ignores cache, venv, IDE files | 10 |
| `README.md` | Project documentation | ~118 |

---

## 5. Dataset details (`Crop_recommendation_1000.csv`)

- **Total rows:** 25,000 data rows (+1 header row).
- **Crops (classes):** 1,000 (each base crop has many cultivar variants, e.g. `almond`, `almond_dwarf_62`, `almond_hybrid_21`).
- **Samples per crop:** 25.
- **8 columns:** `N, P, K, temperature, humidity, ph, rainfall, label`.

### The 7 features (inputs)
| Feature | Meaning | Unit | Valid Range |
| :--- | :--- | :--- | :--- |
| N | Nitrogen | ratio in soil | 0 – 140 |
| P | Phosphorus | ratio in soil | 5 – 145 |
| K | Potassium | ratio in soil | 5 – 205 |
| temperature | Temperature | °C | 8.8 – 43.7 |
| humidity | Relative Humidity | % | 14.0 – 100.0 |
| ph | Soil pH | scale | 3.5 – 9.9 |
| rainfall | Rainfall | mm | 20.0 – 298.6 |

---

## 6. ALL FUNCTIONS — what they do and where they live

### 📄 `data_prep.py`
| Function | What it does |
| :--- | :--- |
| `load_and_preprocess_data(dataset_path, test_size=0.2, random_state=42)` | Finds the CSV, checks all 8 columns exist, drops null rows, splits into X_train/X_test/y_train/y_test using stratified 80/20 split. Returns those 4 sets. |

**Important constants here:**
- `FEATURE_NAMES` — list of the 7 input column names.
- `TARGET_COLUMN` — `"label"` (the answer column).
- `FEATURE_METADATA` — dictionary of label, unit, min, max for each feature (used by GUI hints & CLI prompts).

---

### 📄 `model_trainer.py`
| Function | What it does |
| :--- | :--- |
| `compute_crop_profiles(X, y)` | For each crop, precomputes the **mean vector** (`mu_matrix`), **standard deviation** (`std_matrix`), and **tolerance band**. Stored as NumPy matrices for fast scoring. |
| `train_gaussian_nb(X_train, X_test, y_train, y_test, verbose=True)` | Trains the `GaussianNB` model, attaches crop profiles to it, predicts on test data, computes **accuracy** + **classification report**. Returns (model, accuracy, report). |
| `get_suitability_badge(percentage)` | Converts a score into a tier label: Optimal / High / Good Alternative / Moderate / Low. |
| `format_crop_name(crop_name)` | Turns raw labels like `almond_dwarf_62` into clean names like "Almond". Handles multi-word crops (Black Pepper, Sweet Potato, etc.). |
| `predict_crop(model, input_values, top_k=6)` | **The main prediction function.** Takes 7 inputs, scores ALL 1000 crops with vectorized math, picks top 6 from distinct crop families, builds an insight sentence. Returns a dict with recommended crop, confidence, top candidates, all scores, inputs, and insight. |

**Constant here:** `MULTI_WORD_CROPS` — mapping for crops whose names have two words.

#### The scoring math inside `predict_crop` (important for viva!)
```
z          = |input - crop_mean| / tolerance      # how far off, in tolerance units
feat_match = exp(-0.5 * (z / 1.5)^2)               # Gaussian closeness, 0 to 1, per feature
raw_score  = (0.75 * mean_match + 0.25 * min_match) * 100
```
- Uses a **Gaussian (bell curve) similarity**: the closer your input is to a crop's ideal, the higher the score.
- `0.75 * mean + 0.25 * min` → rewards overall fit **and** penalizes any single badly-wrong feature.
- Done for all 1000 crops in **one NumPy matrix operation** (very fast, <2ms).

**Why not plain Naive Bayes for scoring?**
Because plain GaussianNB multiplies probabilities over 7 features and gives ~99.9% to one crop
and ~0% to the rest — useless for *comparing* crops. So GaussianNB is used for **accuracy/validation**,
and the custom Gaussian suitability scorer is used for the **percentage recommendations**.

---

### 📄 `main.py`
| Function | What it does |
| :--- | :--- |
| `print_banner()` | Prints the ASCII title banner. |
| `print_prediction_result(result)` | Prints the recommendation nicely in the terminal with ASCII bar charts (`████`). |
| `run_interactive_cli(model)` | Prompts the user to type their own 7 values in the terminal. |
| `main()` | The entrypoint. Parses CLI flags, runs the 3-step pipeline (load → train → predict), then launches GUI or CLI. |

**Constant here:** `SAMPLE_INPUT` — a default set of 7 values used for the demo prediction.

**Command-line flags:**
| Command | Effect |
| :--- | :--- |
| `python main.py` | Runs pipeline, then opens the GUI (default) |
| `python main.py --cli` | Terminal only, no GUI |
| `python main.py --interactive` | Asks you to type your own values |
| `python main.py --gui-only` | Opens GUI without printing the full report |

---

### 📄 `gui_app.py`
| Function / Class | What it does |
| :--- | :--- |
| `class CropChartCanvas(tk.Canvas)` | Custom widget that hand-draws a responsive horizontal **bar chart** of the top 6 crops. |
| `CropChartCanvas.set_data(candidates)` | Loads new data and redraws the chart. |
| `CropChartCanvas.redraw()` | Draws grid lines, bars, colors, labels, percentages. |
| `class CropRecommendationApp` | The main GUI application (window, cards, form). |
| `__init__` | Builds the window, loads/trains model, builds all sections. |
| `_setup_styles()` | Defines the color palette and fonts. |
| `_build_header()` | Top title bar. |
| `_build_main_content()` | Left input card + right result/chart card. |
| `_build_footer()` | Bottom status bar. |
| `_apply_preset(name)` | Fills the form with a preset crop's values and predicts. |
| `_clear_entries()` | Clears all inputs. |
| `on_predict()` | Validates inputs, calls `predict_crop`, updates hero card, chart, and insight. |
| `launch_gui(model=None)` | Creates the Tk window and starts the app. |

**Constants here:**
- `PRESETS` — 8 one-click test crops (Rice, Litchi, Watermelon, Mango, Wheat, Cotton, Apple, Coffee).
- `CROP_ICONS` — emoji icons for crops.

#### Suitability tiers (colors in GUI)
| Tier | Score | Color |
| :--- | :--- | :--- |
| Optimal Match | 85–100% | Emerald green |
| High Match | 70–84.9% | Mint green |
| Good Alternative | 55–69.9% | Amber |
| Moderate Match | 40–54.9% | Blue |
| Low Suitability | < 40% | Slate gray |

---

## 7. How the whole system flows (end to end)

```
1. User runs  python gui_app.py  (or python main.py)
2. data_prep.py        → loads 25,000 rows, cleans, splits 80/20
3. model_trainer.py    → trains GaussianNB + builds crop profile matrices
4. User enters 7 values (or clicks a preset)
5. predict_crop()      → scores all 1000 crops in one NumPy pass
6. GUI / CLI shows     → best crop + top 6 alternatives (%) + insight sentence
```

Module dependency (who imports whom):
```
data_prep.py   ← imported by →  model_trainer.py, gui_app.py, main.py
model_trainer.py ← imported by → gui_app.py, main.py
main.py  →  can launch  →  gui_app.py
```

---

## 8. Likely viva questions & short answers

**Q: Which algorithm did you use?**
Gaussian Naive Bayes (scikit-learn) for classification/accuracy, plus a custom Gaussian
similarity scorer for the suitability percentages.

**Q: Why Naive Bayes?**
It's fast, simple, works well with continuous numeric features, and assumes each feature follows
a normal (Gaussian) distribution — which fits soil/climate data.

**Q: What is train_test_split and why 80/20?**
It divides data into a training set (to learn) and a testing set (to check accuracy). 80/20 is a
common ratio that leaves enough data to learn while keeping a fair test. We used `stratify` so
every crop appears in both sets.

**Q: What are the 7 features?**
Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, Rainfall.

**Q: How many crops / records?**
1,000 crops, 25,000 records (25 samples each).

**Q: Why NumPy?**
To score all 1,000 crops at once using vectorized matrix math instead of slow Python loops.

**Q: What does pandas do here?**
Reads the CSV file and holds the data in DataFrames; also groups data by crop to compute averages.

**Q: What is Tkinter?**
Python's built-in library for making desktop GUI applications (windows, buttons, forms).

**Q: What is stratified split?**
A split that keeps the same proportion of each crop in both training and testing sets.

**Q: How do you run it without a display (server)?**
`python main.py --cli` or `python main.py --interactive` (no GUI needed).

---

## 9. Quick facts to memorize

- **Language:** Python 3
- **ML library:** scikit-learn — **Algorithm:** Gaussian Naive Bayes
- **Math library:** NumPy | **Data library:** pandas | **GUI:** Tkinter
- **Dataset:** 25,000 rows, 1,000 crops, 7 features + 1 label, CSV format
- **Split:** 80% train / 20% test, stratified, random_state=42
- **Output:** 1 best crop + top 6 alternatives with % + insight
- **Main prediction function:** `predict_crop()` in `model_trainer.py`
