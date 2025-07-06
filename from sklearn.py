from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
import numpy as np

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    weights=[0.9, 0.1],
    random_state=42
)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print('Önce:', np.bincount(y))
print('Sonra:', np.bincount(y_resampled))

