import math
import warnings
import numpy as np
import pandas as pd
import pipeline


CAT_FEATURES = ['Sex', 'Occupation', 'Marriage_Status', 'credit_grade']


def compute_shap(data: dict, model, explainer) -> dict:
    fitted_transformer = model.named_steps['preprocess']
    feature_names_out = fitted_transformer.get_feature_names_out()

    processed_input = pipeline.preprocess_input_for_shap(data, fitted_transformer)
    processed_df = pd.DataFrame(processed_input, columns=feature_names_out)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            category=UserWarning,
        )
        shap_values = explainer.shap_values(processed_df)

    if isinstance(shap_values, list):
        final_shap_values = shap_values[1][0]
    else:
        final_shap_values = shap_values[0]

    # Aggregate one-hot encoded features back to original names
    aggregated_values = {}
    for name, val in zip(feature_names_out, final_shap_values):
        if "__" not in name:
            parent_name = name
        elif name.startswith("cat__"):
            parent_name = name.split("__")[1]
            for original_cat in CAT_FEATURES:
                if original_cat in name:
                    parent_name = original_cat
                    break
        else:
            parent_name = name.split("__")[1]
        aggregated_values[parent_name] = aggregated_values.get(parent_name, 0) + val

    expected = explainer.expected_value
    if isinstance(expected, (list, tuple, np.ndarray)):
        base_value = float(np.asarray(expected).reshape(-1)[-1])
    else:
        base_value = float(expected)
    shap_sum = sum(aggregated_values.values())
    total_summation = base_value + shap_sum
    probability = 1 / (1 + math.exp(-total_summation))

    return {
        "base_value": base_value,
        "shap_values": {k: round(float(v), 4) for k, v in aggregated_values.items()},
        "log_odds": float(total_summation),
        "probability": round(probability, 4),
    }
