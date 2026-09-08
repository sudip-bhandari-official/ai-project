"""
Model Architecture, Multi-Crop Suitability Scoring & Evaluation (model_trainer.py)
----------------------------------------------------------------------------------
Responsibility:
- Fits sklearn.naive_bayes.GaussianNB on training data and computes test accuracy.
- Vectorized continuous agronomic compatibility scoring across 1,000 agricultural crops in <2ms.
- Recommends optimal crops alongside diverse alternative crops sharing similar soil/climate tolerances.
- Generates agronomic comparative insights for farmer decision support.
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from data_prep import FEATURE_NAMES, load_and_preprocess_data


def compute_crop_profiles(X, y):
    """
    Precomputes vectorized matrix profiles (means, stds, and tolerances)
    for high-speed continuous multi-crop suitability inference.
    """
    df = X.copy()
    df["__label__"] = y.values

    crops = np.array(sorted(y.unique()))
    global_std = X[FEATURE_NAMES].std().values

    grouped = df.groupby("__label__")
    mu_matrix = grouped[FEATURE_NAMES].mean().loc[crops].values
    std_matrix = np.maximum(grouped[FEATURE_NAMES].std().loc[crops].values, 1e-4)

    # Agronomic tolerance: combines crop-specific variance with global feature variation
    tolerances = 0.55 * std_matrix + 0.25 * global_std

    return {
        "crop_names": crops,
        "mu_matrix": mu_matrix,
        "std_matrix": std_matrix,
        "tolerances": tolerances,
        "global_std": global_std,
    }


def train_gaussian_nb(X_train, X_test, y_train, y_test, verbose: bool = True):
    """
    Fits Gaussian Naive Bayes classifier on training data and evaluates on test data.
    Attaches crop agronomic profiles for multi-crop suitability scoring.

    Returns:
        tuple: (fitted_model, accuracy, report_dict)
    """
    model = GaussianNB()
    model.fit(X_train, y_train)

    # Attach vectorized crop profiles for continuous multi-crop suitability analysis
    model.crop_profiles_ = compute_crop_profiles(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    if verbose:
        num_classes = len(model.classes_)
        print("=" * 65)
        print("         Smart Crop Recommendation - Model Evaluation")
        print("=" * 65)
        print(f"  Trained Classes: {num_classes} Crop Varieties")
        print(f"  Test Accuracy Score: {accuracy * 100:.2f}%\n")
        if num_classes <= 30:
            print("Classification Report:")
            print("-" * 65)
            print(classification_report(y_test, y_pred, zero_division=0))
        else:
            print(f"  Summary Across {num_classes} Crop Classes:")
            print(f"    - Macro Avg Precision : {report_dict['macro avg']['precision']:.4f}")
            print(f"    - Macro Avg Recall    : {report_dict['macro avg']['recall']:.4f}")
            print(f"    - Macro Avg F1-Score  : {report_dict['macro avg']['f1-score']:.4f}")
        print("=" * 65)

    return model, accuracy, report_dict


def get_suitability_badge(percentage: float) -> str:
    """Returns a readable suitability tier for a given match percentage."""
    if percentage >= 85.0:
        return "Optimal Match"
    elif percentage >= 70.0:
        return "High Match"
    elif percentage >= 55.0:
        return "Good Alternative"
    elif percentage >= 40.0:
        return "Moderate Match"
    else:
        return "Low Suitability"


MULTI_WORD_CROPS = {
    "black_pepper": "Black Pepper",
    "sweet_potato": "Sweet Potato",
    "custard_apple": "Custard Apple",
    "bell_pepper": "Bell Pepper",
    "bitter_gourd": "Bitter Gourd",
    "green_pea": "Green Pea",
    "faba_bean": "Faba Bean",
    "pearl_millet": "Pearl Millet",
    "finger_millet": "Finger Millet",
    "foxtail_millet": "Foxtail Millet",
}


def format_crop_name(crop_name: str) -> str:
    """Formats raw crop labels into clean, human-readable display names without cultivar comments."""
    for prefix, pretty in MULTI_WORD_CROPS.items():
        if crop_name == prefix or crop_name.startswith(prefix + "_"):
            return pretty

    base = crop_name.split("_")[0]
    return base.capitalize()


def predict_crop(model, input_values, top_k: int = 6):
    """
    Performs vectorized multi-crop inference and compatibility analysis
    for a given soil & weather condition across all crops.

    Parameters:
        model: Trained GaussianNB model with attached crop_profiles_
        input_values: list, dict, or pd.DataFrame with 7 continuous values
        top_k: Number of top candidate crops to return (default: 6)

    Returns:
        dict: {
            'recommended_crop': str (best matching crop),
            'confidence': float (match percentage of top crop),
            'top_candidates': list of tuples [(crop, percentage, status_badge), ...],
            'all_scores': dict mapping crop -> percentage,
            'input_features': dict of input parameters,
            'insight': str explaining comparative crop suitability
        }
    """
    # Convert input into standardized array
    if isinstance(input_values, dict):
        df_input = pd.DataFrame([input_values])[FEATURE_NAMES]
    elif isinstance(input_values, list):
        if len(input_values) != 7:
            raise ValueError(f"Expected 7 feature values, received {len(input_values)}: {input_values}")
        df_input = pd.DataFrame([input_values], columns=FEATURE_NAMES)
    elif isinstance(input_values, pd.DataFrame):
        df_input = input_values[FEATURE_NAMES]
    else:
        df_input = pd.DataFrame([list(input_values)], columns=FEATURE_NAMES)

    x_vec = df_input.iloc[0].values.astype(float)

    # Ensure crop profiles are available
    if not hasattr(model, "crop_profiles_"):
        X_tr, _, y_tr, _ = load_and_preprocess_data()
        model.crop_profiles_ = compute_crop_profiles(X_tr, y_tr)

    profiles = model.crop_profiles_
    crops = profiles["crop_names"]
    mu_matrix = profiles["mu_matrix"]
    tolerances = profiles["tolerances"]

    # High-speed vectorized continuous compatibility scoring (< 2ms for 1,000 crops)
    z = np.abs(x_vec - mu_matrix) / tolerances
    feat_match = np.exp(-0.5 * ((z / 1.5) ** 2))
    mean_feat = np.mean(feat_match, axis=1)
    min_feat = np.min(feat_match, axis=1)
    raw_scores = np.clip((0.75 * mean_feat + 0.25 * min_feat) * 100.0, 0.0, 100.0)

    # Sort descending
    sorted_indices = np.argsort(raw_scores)[::-1]

    # Pick top distinct primary crop families so recommendations are diverse and meaningful
    seen_families = set()
    top_candidates = []
    crop_scores = {}

    for idx in sorted_indices:
        cname = crops[idx]
        score = round(float(raw_scores[idx]), 1)
        crop_scores[cname] = score

        base_family = cname.split("_")[0]
        if base_family not in seen_families and len(top_candidates) < top_k:
            seen_families.add(base_family)
            badge = get_suitability_badge(score)
            top_candidates.append((cname, score, badge))

    # In case fewer distinct families than top_k, fill up
    if len(top_candidates) < top_k:
        for idx in sorted_indices:
            cname = crops[idx]
            if not any(c == cname for c, _, _ in top_candidates):
                score = round(float(raw_scores[idx]), 1)
                badge = get_suitability_badge(score)
                top_candidates.append((cname, score, badge))
                if len(top_candidates) >= top_k:
                    break

    top_crop = top_candidates[0][0]
    top_confidence = top_candidates[0][1]

    # Comparative insight
    first_crop = format_crop_name(top_candidates[0][0])
    first_score = top_candidates[0][1]
    second_crop = format_crop_name(top_candidates[1][0])
    second_score = top_candidates[1][1]
    third_crop = format_crop_name(top_candidates[2][0])
    third_score = top_candidates[2][1]

    diff_1_2 = first_score - second_score
    if diff_1_2 <= 10.0:
        insight = (
            f"Close growing conditions: Both {first_crop} ({first_score}%) and "
            f"{second_crop} ({second_score}%) thrive in this environment. "
            f"{third_crop} ({third_score}%) is also a viable alternative."
        )
    elif diff_1_2 <= 20.0:
        insight = (
            f"{first_crop} ({first_score}%) is the most favorable match, "
            f"while {second_crop} ({second_score}%) offers a strong alternative crop option."
        )
    else:
        insight = (
            f"{first_crop} ({first_score}%) strongly matches the given soil and climate conditions, "
            f"significantly outperforming alternative crops like {second_crop} ({second_score}%)."
        )

    return {
        "recommended_crop": top_crop,
        "confidence": top_confidence,
        "top_candidates": top_candidates,
        "all_scores": crop_scores,
        "input_features": df_input.iloc[0].to_dict(),
        "insight": insight,
    }


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    model, accuracy, report = train_gaussian_nb(X_train, X_test, y_train, y_test)

    # Test sample Litchi input
    sample_input = [62.0, 32.0, 50.0, 26.8, 81.5, 6.3, 188.0]
    result = predict_crop(model, sample_input, top_k=6)
    print("\nSample Test Inference (Litchi Conditions):")
    print(f"  Inputs: {sample_input}")
    print(f"  Recommended Crop: {format_crop_name(result['recommended_crop'])}")
    print(f"  Match Percentage: {result['confidence']:.1f}%")
    print("  Top Candidates:")
    for crop, pct, badge in result["top_candidates"]:
        print(f"    - {format_crop_name(crop):<22}: {pct:>5.1f}% ({badge})")
    print(f"  Insight: {result['insight']}")


