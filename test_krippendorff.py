import numpy as np
import krippendorff
from krippendorff import alpha

# Dataset1, Annotator1
#annotator1_annotations = [5, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 5, 0, 5, 0, 5, 0, 3, 0, 0, 5, 3, 0, 0, 4, 3, 0, 4, 0, 0, 0, 0, 0, 5, 0, 3, 0, 3, 0, 4, 5, 0, 5, 4, 0, 0, 4, 0, 0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 5, 0, 5, 4, 0, 0, 0, 4, 5, 0, 0, 4, 4, 0, 5, 5, 5, 0, 5, 0, 5, 0, 5, 5, 4, 5, 5, 0, 0, 2, 0, 5, 5, 4, 3, 5, 5, 0, 0, 5, 5, 5, 5, 0, 0, 0, 0, 0, 0, 5]
#annotator1_confidence = [5, 4, 3, 3, 5, 3, 3, 3, 5, 5, 5, 5, 3, 4, 5, 3, 4, 3, 0, 5, 3, 5, 0, 3, 3, 3, 3, 4, 4, 4, 5, 5, 0, 0, 5, 5, 4, 5, 3, 5, 5, 3, 0, 4, 4, 0, 5, 3, 5, 5, 5, 0, 4, 5, 4, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 4, 5, 5, 4, 5, 4, 4, 5, 5, 5, 0, 4, 5, 5, 5, 4, 5, 4, 5, 4, 5, 4, 4, 4, 5, 4, 4, 4, 4, 5, 5, 5, 4, 4, 4, 5, 4, 5, 0, 5, 4, 4, 5]
# --------------------
# Dataset2, Annotator 1
#annotator1_annotations = [4, 0, 1, 5, 0, 0, np.nan, 4, np.nan, 2, 0, 5, 5, 0, 0, 0, 0, 5, 5, 5, 5, 5, 4, np.nan, 5, 0, 0, 0, 4, 5, 0, 0, 4, 5, 5, 5, 1, 0, 0, 0, 4, 1, 0, 0, 0, 0, 5, 0, 0, 5, 5, 0, 5, 4, 5, 5, 0, 0, 0, 0, 0, 4, 0, 0, 0, 5, 0, 0, 5, 5, 0, 0, 0, 5, 4, 0, 0, 5, 5, 0, 3, 4, 0, 5, 0, 5, 0, 0, 0, 4, 4, 4, 0, 0, 2, 5, 5, 0, 0, 5, 0, 5, 0, 5, 5, 0, 5, 5, 0]
#annotator1_confidence = [3, 1, 3, 4, 5, 5, np.nan, 4, np.nan, 4, 4, 5, 4, 5, 4, 3, 0, 5, 5, 5, 4, 4, 4, np.nan, 5, 2, 2, 5, 4, 4, 4, 4, 4, 4, 5, 5, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4, 5, 4, 4, 5, 4, 5, 3, 4, 5, 5, 5, 4, 0, 4, 4, 4, 4, 4, 5, 0, 3, 4, 4, 0, 0, 4, 4, 3, 4, 0, 5, 5, 5, 3, 2, 5, 5, 4, 4, 5, 4, 4, 4, 4, 4, 4, 5, 1, 5, 4, 4, 5, 0, 5, 5, 4, 5, 3, 4, 4, 3, 0]
# --------------------
# Dataset 3, Annotator 1
#annotator1_annotations = [3, np.nan, 0, 2, 5, 0, 0, 1, 0, 4, 0, 3, 1, 0, 0, 0, 0, 0, 0, 4, np.nan, 0, 4, 0, 0, 4, 0, 0, 5, 1, 0, 0, 0, 0, 0, 0, 3, 5, 0, 5, 0, 4, 0, 0, 0, 5, 5, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 4, 4, 0, 4, 0, 5, 0, 0, 5, 0, 5, 0, 0, 0, 4, 3, 0, 5, 5, 0, 0, 0, 0, 0, 0, 5, 0, 0, 3, 0, 0, 0, 4, 0, 5, 0, 5, 0, np.nan, 3, 0, 4, 2, 5, 0, 4, 5, 4, 0, 3, 3, 4]
#annotator1_confidence = [4, np.nan, 4, 4, 5, 4, 4, 4, 4, 4, 5, 3, 4, 3, 4, 4, 5, 5, 5, 5, np.nan, 5, 4, 4, 5, 4, 5, 5, 5, 4, 0, 5, 5, 5, 1, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 5, 3, 5, 5, 5, 4, 5, 4, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 4, 4, 5, 5, 5, 5, 5, 5, 5, 4, 5, 5, 5, 5, 3, 5, 5, 3, 4, 5, 5, 5, 4, 3, np.nan, 3, 5, 4, 3, 5, 5, 5, 5, 4, 5, 3, 2, 5]
# --------------------
# Dataset4, Annotator 1
annotator1_annotations = [0, 4, 2, 0, 2, 4, np.nan, 4, 3, 0, 5, 2, 0, 2, 5, 0, 4, 2, 5, 0, 2, 5, 5, 0, 0, 4, 0, 0, 4, 1, 3, 4, 0, 0, 3, 4, 0, 0, 3, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 5, 0, 5, 1, 0, 0, 0, 4, 0, 5, 4, 0, 3, 0, 0, 0, 0, 5, 4, 1, 4, 5, 0, 4, 5, 0, 0, 5, 0, 1, 0, 0, 4, 3, 0, 0, 0, 0, 0, 0, 0, 5, 0, 4, 0, 0, 5, 5, 0, 0, 0, 0, 0, 5, np.nan, 0, 5, 0, 5, 0, 5]
annotator1_confidence = [5, 4, 4, 5, 3, 5, np.nan, 5, 2, 0, 5, 4, 5, 3, 5, 5, 5, 3, 5, 5, 1, 5, 5, 5, 5, 4, 5, 5, 4, 1, 2, 4, 5, 5, 2, 3, 5, 5, 3, 5, 5, 5, 5, 4, 4, 5, 5, 5, 5, 5, 5, 5, 3, 5, 5, 5, 4, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 4, 4, 4, 5, 5, 4, 5, 5, 5, 5, 5, 3, 5, 5, 4, 3, 5, 5, 5, 5, 3, 5, 5, 5, 5, 4, 5, 5, 0, 5, 5, 5, 5, 5, 4, 5, np.nan, 4, 5, 5, 5, 5, 5]

# Calculate the average of non-missing values
average_value1 = np.nanmean(annotator1_annotations)
average_value2 = np.nanmean(annotator1_confidence)

# Replace missing values with the average value
imputed_annotations = [average_value1 if np.isnan(val) else val for val in annotator1_annotations]
imputed_confidence = [average_value2 if np.isnan(val) else val for val in annotator1_confidence]
print("annotation:", imputed_annotations)
print("confidencde:", imputed_confidence)

# Dataset2, Annotator2
#annotator2_annotations = [3, 5, 0, 4, 1, 0, 0, 3, 0, 0, 0, 5, 5, 0, 1, 0, 5, 4, 4, 5, 4, 1, 0, 5, 4, 0, 0, 0, 3, 3, 0, 0, 4, 4, 5, 5, 2, 0, 0, 0, 3, 0, 0, 1, 0, 0, 1, 0, 0, 0, 3, 0, 3, 0, 4, 1, 0, 0, 0, 3, 0, 4, 0, 0, 0, 5, 4, 0, 3, 2, 0, 0, 0, 4, 1, 0, 0, 3, 4, 0, 4, 5, 0, 5, 0, 4, 0, 0, 0, 5, 4, 2, 0, 0, 5, 5, 4, 0, 0, 5, 0, 5, 0, 5, 5, 0, 0, 5, 0]
#annotator2_confidence = [4, 4, 3, 3, 3, 4, 0, 3, 4, 1, 4, 5, 5, 5, 3, 1, 3, 3, 4, 4, 4, 3, 1, 4, 3, 4, 2, 4, 2, 3, 4, 3, 4, 4, 4, 4, 3, 4, 1, 0, 3, 2, 4, 1, 3, 3, 2, 2, 3, 1, 3, 4, 2, 0, 3, 3, 5, 5, 4, 3, 4, 2, 4, 2, 4, 5, 4, 4, 3, 2, 2, 3, 5, 2, 3, 4, 4, 3, 3, 5, 3, 4, 4, 5, 5, 4, 4, 3, 4, 5, 3, 1, 3, 4, 5, 5, 2, 4, 4, 4, 4, 4, 5, 5, 5, 4, 0, 4, 2]
# --------------------------------
# Dataset3, Annotator2
#annotator2_annotations = [4, 0, 0, 2, 5, 0, 0, 5, 0, 4, 0, 5, 3, 0, 3, 2, 0, 0, 0, 4, 0, 0, 5, 2, 3, 0, 2, 1, 3, 4, 0, 0, 0, 0, 4, 0, 4, 5, 5, 5, 0, 5, 0, 0, 0, 5, 5, 1, 0, 0, 4, 0, 4, 0, 0, 0, 3, 5, 4, 0, 4, 0, 5, 0, 0, 4, 0, 5, 0, 0, 0, 1, 2, 0, 3, 3, 0, 0, 1, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 4, 0, 5, 4, 5, 0, 2, 4, 4, 0, 0, 0, 4]
#annotator2_confidence = [4, 4, 4, 3, 5, 4, 5, 4, 3, 4, 3, 4, 4, 0, 3, 3, 4, 4, 2, 0, 5, 4, 4, 2, 4, 0, 4, 3, 3, 4, 0, 4, 4, 4, 3, 4, 4, 4, 5, 4, 4, 4, 3, 4, 4, 5, 3, 4, 0, 4, 4, 3, 3, 3, 4, 4, 3, 5, 4, 4, 3, 4, 5, 4, 4, 4, 4, 5, 3, 4, 3, 2, 3, 4, 3, 3, 4, 4, 4, 4, 2, 4, 4, 4, 4, 0, 4, 4, 0, 0, 4, 4, 3, 2, 2, 4, 4, 4, 5, 3, 4, 4, 3, 3, 4, 4, 0, 0, 3]
# --------------------------------
# Dataset4, Annotator2
annotator2_annotations = [0, 2, 2, 0, 3, 4, 0, 0, 5, 0, 4, 1, 0, 3, 3, 0, 1, 0, 5, 0, 4, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 3, 0, 4, 2, 0, 1, 0, 0, 0, 0, 3, 1, 0, 2, 2, 0, 3, 2, 0, 0, 5, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 1, 0, 0, 5, 0, 3, 0, 0, 2, 3, 0, 0, 0, 0, 1, 3, 3, 2, 1, 0, 5, 0, 3]
annotator2_confidence = [5, 3, 3, 5, 3, 5, 5, 4, 5, 5, 4, 2, 5, 2, 3, 5, 2, 3, 5, 5, 5, 3, 1, 5, 5, 2, 3, 5, 2, 5, 4, 5, 5, 5, 5, 5, 4, 5, 3, 5, 5, 5, 5, 2, 5, 5, 5, 5, 5, 3, 5, 5, 4, 5, 5, 5, 3, 5, 4, 5, 5, 2, 5, 5, 5, 4, 4, 2, 3, 3, 3, 5, 2, 4, 5, 5, 5, 2, 4, 5, 5, 3, 1, 5, 5, 5, 5, 2, 5, 4, 4, 5, 2, 5, 5, 2, 4, 5, 5, 5, 5, 2, 3, 2, 2, 1, 5, 5, 5, 3]

# # Calculate the average of non-missing values
# average_value1 = np.nanmean(annotator2_annotations)
# average_value2 = np.nanmean(annotator2_confidence)
#
# # Replace missing values with the average value
# imputed_annotations = [average_value1 if np.isnan(val) else val for val in annotator2_annotations]
# imputed_confidence = [average_value2 if np.isnan(val) else val for val in annotator2_confidence]
# print("annotation:", imputed_annotations)
# print("confidencde:", imputed_confidence)

# Dataset2, Annotator3
#annotator3_annotations = [3, 4, 0, 5, 0, 4, 3, 5, 3, 3, 5, 5, 0, 4, 0, 4, 4, 5, 2, 5, 5, 5, 5, 5, 5, 0, 3, 0, 5, 5, 0, 0, 5, 5, 5, 5, 3, 0, 0, 4, 5, 5, 0, 5, 0, 3, 5, 5, 0, 4, 5, 0, 5, 0, 5, 5, 0, 0, 0, 3, 0, 5, 4, 4, 4, 5, 4, 4, 5, 5, 0, 0, 0, 5, 4, 0, 0, 5, 3, 0, 5, 5, 0, 5, 4, 5, 0, 0, 0, 5, 5, 4, 0, 0, 5, 5, 5, 5, 0, 5, 5, 5, 0, 5, 5, 3, 0, 5, 0]
#annotator3_confidence = [3, 3, 4, 4, 4, 3, 4, 5, 3, 3, 4, 4, 3, 3, 1, 3, 4, 5, 1, 5, 4, 5, 3, 4, 5, 0, 2, 4, 4, 4, 0, 2, 4, 5, 5, 5, 2, 4, 2, 3, 5, 5, 4, 4, 2, 4, 4, 3, 3, 4, 5, 4, 3, 0, 5, 3, 3, 5, 4, 4, 4, 4, 4, 5, 4, 5, 2, 3, 4, 3, 4, 5, 4, 4, 4, 2, 3, 4, 2, 5, 3, 3, 4, 5, 2, 4, 5, 4, 4, 4, 3, 3, 2, 3, 3, 4, 4, 3, 3, 4, 4, 4, 4, 4, 4, 4, 0, 4, 2]
# -------------------------------------
# Dataset3, Annotator3
#annotator3_annotations = [4, 0, 0, 0, 4, 2, 4, 5, 3, 4, 0, 5, 4, 4, 4, 4, 0, 0, 1, 5, 0, 0, 5, 0, 3, 4, 1, 0, 5, 4, 0, 0, 0, 0, 0, 0, 5, 5, 4, 5, 0, 4, 0, 0, 0, 5, 5, 0, 5, 0, 5, 0, 3, 0, 0, 0, 1, 5, 4, 0, 4, 0, 5, 0, 0, 4, 0, 5, 0, 0, 0, 1, 1, 0, 5, 5, 0, 0, 0, 0, 2, 0, 5, 0, 1, 4, 0, 1, 1, 0, 0, 5, 4, 0, 4, 0, 3, 0, 5, 4, 5, 0, 3, 5, 5, 0, 5, 4, 4]
#annotator3_confidence = [5, 5, 5, 4, 2, 4, 3, 5, 3, 5, 5, 5, 3, 2, 3, 4, 5, 5, 5, 4, 5, 5, 4, 0, 2, 0, 5, 5, 5, 4, 5, 5, 5, 5, 5, 5, 4, 5, 4, 4, 5, 2, 5, 5, 5, 5, 4, 5, 5, 5, 5, 5, 1, 5, 5, 5, 4, 5, 4, 5, 3, 5, 5, 5, 5, 2, 5, 4, 5, 5, 5, 4, 4, 5, 5, 4, 5, 5, 5, 5, 4, 5, 4, 5, 0, 2, 5, 4, 0, 0, 5, 4, 4, 0, 1, 5, 1, 5, 4, 4, 5, 5, 1, 5, 5, 5, 4, 0, 0]
# -------------------------------------
# Dataset4, Annotator3
annotator3_annotations = [5, 0, 4, 0, 0, 5, 0, 5, 5, 1, 5, 2, 1, 0, 4, 1, 5, 3, 5, 2, 0, 5, 1, 1, 1, 3, 3, 1, 4, 2, 2, 0, 3, 1, 3, 5, 2, 0, 2, 2, 0, 0, 3, 3, 5, 0, 3, 0, 0, 4, 0, 5, 2, 0, 0, 0, 5, 0, 5, 5, 1, 3, 0, 0, 0, 0, 5, 5, 5, 4, 5, 0, 2, 5, 0, 1, 5, 1, 0, 0, 0, 4, 4, 0, 1, 0, 0, 0, 0, 1, 5, 0, 5, 0, 0, 0, 5, 0, 0, 0, 0, 1, 4, 5, 4, 4, 0, 5, 0, 4]
annotator3_confidence = [5, 0, 3, 0, 0, 5, 0, 5, 5, 5, 5, 3, 3, 0, 3, 2, 5, 3, 3, 1, 0, 5, 1, 2, 2, 3, 4, 4, 3, 5, 4, 4, 4, 2, 5, 4, 3, 0, 2, 2, 0, 0, 3, 4, 5, 5, 4, 5, 5, 4, 5, 4, 4, 5, 5, 5, 5, 5, 5, 4, 4, 4, 5, 5, 5, 4, 5, 4, 4, 4, 4, 5, 4, 5, 5, 4, 5, 4, 4, 5, 5, 4, 4, 5, 5, 4, 5, 0, 5, 5, 5, 5, 5, 5, 5, 0, 5, 4, 4, 4, 5, 2, 4, 3, 4, 3, 4, 4, 5, 4]

# annotations_matrix = np.array([
#     annotator1_annotations,
#     imputed_annotations,
#     annotator3_annotations
# ])
# confidence_matrix = np.array([
#     annotator1_confidence,
#     imputed_confidence,
#     annotator3_confidence
# ])

annotations_matrix = np.array([
    imputed_annotations,
    annotator2_annotations,
    annotator3_annotations
])
confidence_matrix = np.array([
    imputed_confidence,
    annotator2_confidence,
    annotator3_confidence
])

alpha_result = alpha(annotations_matrix, level_of_measurement='ordinal')
weighted_alpha_result = np.mean(np.square(confidence_matrix) * alpha_result)


# weighted_agreements = []
# for item_idx in range(annotations_matrix.shape[1]):
#     agreement = krippendorff.alpha(np.expand_dims(annotations_matrix[:, item_idx], axis=1))
#     confidence_scores = confidence_matrix[:, item_idx]
#     weighted_agreement = np.mean(np.square(confidence_scores) * agreement)
#     weighted_agreements.append(weighted_agreement)

#weighted_krippendorff_alpha = np.mean(weighted_agreements)


print("Krippendorff's Alpha (with Confidence):", alpha_result)
print("Weighted Alpha (with Confidence):", weighted_alpha_result)