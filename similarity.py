import numpy as np

def find_most_similar(needle, haystack):
    needle_norm = np.linalg.norm(needle)
    similarity_scores = [
        np.dot(needle, item) / (needle_norm * np.linalg.norm(item)) for item in haystack
    ]
    return sorted(zip(similarity_scores, range(len(haystack))), reverse=True)

needle = np.array([1, 2, 3])
haystack = [
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([1, 2, 3])
]

print(find_most_similar(needle, haystack))