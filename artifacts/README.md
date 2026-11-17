# Saved model artifacts

`model_registry.json` describes all available models, their preprocessing, validation-selected shared thresholds, fresh validation metrics, training configuration, and the Python package versions used to save them.

- `preprocessing/tfidf_vectorizer.joblib`: training-only 1,000-term unigram TF-IDF vectorizer.
- `preprocessing/pca_704.joblib`: training-only PCA used by U2-U5.
- `models/*.joblib`: scikit-learn classifiers for U2-U6.
- `models/*.pt`: PyTorch state dictionaries and the architecture/training settings for U7 and the four U8 MLP search runs.

Use `letterboxd_project.predict`; its default threshold for each model comes from the validation partition. Never load joblib files from an untrusted source.
