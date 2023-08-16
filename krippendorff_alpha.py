import numpy as np

def krippendorffs_alpha_with_confidence_and_missing_values(hatefulnes_scores, confidence_scores):
  """Calculates Krippendorff's alpha with confidence score when there are missing values.

  Args:
    hatefulnes_scores: A list of hatefulness scores.
    confidence_scores: A list of confidence scores.

  Returns:
    The Krippendorff's alpha with confidence score.
  """

  length = len(hatefulnes_scores)
  if length != len(confidence_scores):
    raise ValueError("The length of the hatefulnes_scores and confidence_scores lists must be the same.")

  # Fill in the missing confidence scores with the mean value of the existing confidence scores.
  mean_confidence_score = np.mean(confidence_scores[confidence_scores != -1])
  for i in range(length):
    if confidence_scores[i] == -1:
      confidence_scores[i] = mean_confidence_score

  non_missing_scores = []
  non_missing_confidence_scores = []
  for i in range(length):
    if hatefulnes_scores[i] != -1 and confidence_scores[i] != -1:
      non_missing_scores.append(hatefulnes_scores[i])
      non_missing_confidence_scores.append(confidence_scores[i])

  weighted_hatefulness_scores = np.multiply(non_missing_scores, non_missing_confidence_scores)
  disagreement = np.sum(np.not_equal(weighted_hatefulness_scores[0:-1], weighted_hatefulness_scores[1:]))
  possible_disagreement = np.sum(np.not_equal(non_missing_scores[0:-1], non_missing_scores[1:]))
  return 1 - (disagreement / possible_disagreement)

if __name__ == "__main__":
  hatefulnes_scores = [3, 2, 1, -1]
  confidence_scores = [4, 3, 2, -1]
  try:
    print(krippendorffs_alpha_with_confidence_and_missing_values(hatefulnes_scores, confidence_scores))
  except ValueError as e:
    print(e)
