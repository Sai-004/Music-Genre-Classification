# Music-Genre-Classification

Welcome to the **Music Genre Classification** project repository! This project explores machine learning techniques to classify music clips into genres (e.g., jazz, rock, classical) using the GTZAN dataset. It includes a comprehensive Jupyter notebook implementing the classification pipeline and a blog-style notebook explaining the project, its connection to multimodal learning, and key insights.

## Project Overview

The GTZAN dataset contains 1000 audio clips (30 seconds each) across 10 genres, with precomputed audio features like Mel-frequency cepstral coefficients (MFCCs), tempo, and spectral centroid. The project:
- Loads and visualizes the dataset to understand feature distributions.
- Preprocesses data using StandardScaler and PCA for dimensionality reduction.
- Trains a voting classifier (XGBoost + Random Forest) to predict genres, achieving an F1-score of 0.78.
- Evaluates performance with metrics, confusion matrices, and feature importance analysis.
- Tests the model on a sample audio file.
- Discusses extensions to multimodal learning by incorporating lyrics or visual features.

The blog notebook provides a narrative explanation, connecting the project to multimodal learning trends and reflecting on learnings and future improvements.

## Prerequisites

To run the notebooks, you need the following:

- **Python**: Version 3.11 or higher
- **Dependencies**: Install required libraries using:
  ```bash
  pip install numpy pandas matplotlib seaborn scikit-learn xgboost librosa ipython joblib
  ```
- **Dataset**: The GTZAN dataset is available on Kaggle:
  - [GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
  - Or download `Data` folder from this repo.
  - Place the dataset in the `data/` directory or update the file paths in the notebooks.
- **Jupyter Notebook**: Install Jupyter to run the `.ipynb` files:
  ```bash
  pip install jupyter
  ```

## Repository Structure

```
music-genre-classification/
├── data/
│   └── (Place GTZAN dataset here: features_30_sec.csv and genres/)
├── music-genre-classification-blog.ipynb          # Main project notebook
├── models/
│   └── music_genre_classifier.pkl                    # Saved voting classifier model
└── README.md                                         # This file
```

- **notebooks/**: Contains the main project notebook with data exploration, preprocessing, modeling, and evaluation.
- **models/**: Stores the trained model (`music_genre_classifier.pkl`) for further use.
- **data/**: Directory for the GTZAN dataset.

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/music-genre-classification.git
   cd music-genre-classification
   ```

2. **Download the GTZAN Dataset**:
   - Download from [Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).
   - Place `features_30_sec.csv` and the `genres_original/` folder in the `data/` directory.

3. **Run the Notebooks**:
   - Start Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Open `music-genre-classifier.ipynb` for the main preoject.
   - Update file paths in the notebooks if your dataset is in a different location.

## Usage

### Running the Main Notebook
- Open `music-genre-classifier.ipynb` in Jupyter.
- Execute cells sequentially to:
  - Load and visualize the dataset (e.g., genre distribution, feature box plots, PCA scatter plots).
  - Preprocess data (scaling, PCA).
  - Train and evaluate the voting classifier.
  - Test the model on a sample audio file (`jazz.00075.wav`).
- The notebook saves the trained model as `music_genre_classifier.pkl`.

### Loading the Saved Model
To use the saved model for predictions:
```python
import joblib
import numpy as np

# Load the model
model = joblib.load('models/music_genre_classifier.pkl')

# Example: Predict on preprocessed features
# audio_processed = <your_preprocessed_features>
# pred_label = model.predict(audio_processed)
```

## Key Features

- **Data Exploration**: Visualizations of genre distributions, feature correlations, and PCA-based genre separability.
- **Preprocessing**: StandardScaler for normalization, PCA for reducing MFCC dimensions.
- **Modeling**: Voting classifier combining XGBoost and Random Forest for robust predictions.
- **Evaluation**: Classification reports, confusion matrices, and feature importance analysis.
- **Blog Narrative**: A detailed explanation connecting the project to multimodal learning and reflecting on learnings.

## Results

- The voting classifier achieves an F1-score of 0.78, excelling at genres like classical and hip-hop but struggling with country and reggae due to feature overlap.
- The model correctly predicts the genre of a sample jazz audio file, demonstrating practical applicability.
- Feature importance analysis highlights the role of MFCC PCA components and spectral features.

## Future Work

- **Multimodal Integration**: Incorporate lyrics (via NLP) or album art (via CNNs) to improve classification.
- **Deep Learning**: Experiment with CNNs or transformers on raw spectrograms.
- **Larger Datasets**: Test on datasets like FMA or AudioSet for better generalizability.
- **Advanced Tuning**: Perform extensive hyperparameter optimization.

## References

- GTZAN Dataset: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
- Librosa: https://librosa.org/
- Scikit-learn: https://scikit-learn.org/
- XGBoost: https://xgboost.readthedocs.io/


## Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or feature requests.
- Submit pull requests with improvements (e.g., new models, visualizations, or multimodal features).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

*Built with ❤️ by M Sai Srinivas*  
*Last Updated: May 9, 2025*