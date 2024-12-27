# NER4CYBER
[MA513] - Hands on Machine Learning for Cybersecurity Project (Named Entity Recognition)

This repository contains a Named Entity Recognition (NER) project designed for identifying and classifying cybersecurity-related entities within text. The system is built using a fine-tuned BERT model and includes scripts for data preparation, model training, evaluation, and prediction.

## Features
- Processes datasets in JSON Lines format.
- Fine-tunes a BERT model for NER tasks in the cybersecurity domain.
- Handles class imbalance through weighted loss.
- Provides utilities for training, validation, and test predictions.
- Validates structural and consistency rules of IOB2 tagging.

---

## Setup Instructions

### Prerequisites
1. Python 3.8 or later.
2. Git installed on your system.

### Installation
Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/ner-cybersecurity.git
   cd ner-cybersecurity
   ```
---

## Dataset Preparation

Ensure the following JSON Lines files are placed in the `data/` directory:
- `NER-TRAINING.jsonlines`: Training dataset.
- `NER-VALIDATION.jsonlines`: Validation dataset.
- `NER-TESTING.jsonlines`: Test dataset.

The JSON Lines format should follow this structure:
```json
{
    "unique_id": <int>,
    "tokens": ["token1", "token2", ...],
    "ner_tags": ["O", "B-Entity", "I-Entity", ...]
}
```

---

## How to Run the Project

### 1. Preprocessing Data
Before training, preprocess the data for tokenization and alignment:
Ensure the file paths (training, validation and testing) are correct and run `DataProcessing.py`.

### 2. Training the Model
Train the NER model using the prepared dataset:
Ensure the new model path is correct and run `Model.py`.

### 3. Generating Predictions on Test Data
Run the model on the test dataset and save predictions to a JSON Lines file:
Ensure the new or old model and output file paths are correct and run `Prediction.py`.

### 4. Postprocessing and Validation
Validate and correct the predictions:
First, ensure the input file (output prediction) path and run `ValidationStructureConsistency.py` to verify the structure and consistency before postprocessing.
Then, ensure the input (output prediction) and output (corrected output prediction) paths are correct and run `PostProcessing.py`.
Finally, rerun `ValidationStructureConsistency.py` and observe if there's any remaining issue.

---

## Repository Structure
```
.
├── corrected_predictions_output/             # Directory for corrected predictions output files.
├── data/                                     # Directory for raw datasets.
├── model/                                    # Directory for trained model outputs.
├── predictions_output/                       # Directory for predictions output files.
├── scripts/                                  # Directory containing utility scripts.
│   ├── DataProcessing.py                     # Script for data preprocessing.
│   ├── Model.py                              # Script for model training.
│   ├── NERDataset.py                         # Script for dataset evaluation.
│   ├── PostProcessing.py                     # Script for postprocessing correction.
│   ├── Prediction.py                         # Script for predictions.
│   ├── ValidationStructureConsistency.py     # Script for predictions validations.
├── README.md                                 # Project documentation.
```

---

## Future Improvements
- Enhance model performance for rare tags (e.g., `I-Modifier`).
- Explore alternative transformer architectures like RoBERTa or ELECTRA.
- Introduce error analysis for class-level insights.
- Add support for nested and overlapping entities.

---



