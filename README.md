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
```bash
python preprocess.py --data_dir data --output_dir processed_data
```

### 2. Training the Model
Train the NER model using the prepared dataset:
```bash
python train.py --train_file processed_data/NER-TRAINING.jsonlines \
                --val_file processed_data/NER-VALIDATION.jsonlines \
                --output_dir model_output \
                --epochs 3 --batch_size 16 --learning_rate 1e-5
```

### 3. Evaluating the Model
Evaluate the trained model on the validation dataset:
```bash
python evaluate.py --model_dir model_output \
                   --val_file processed_data/NER-VALIDATION.jsonlines
```

### 4. Generating Predictions on Test Data
Run the model on the test dataset and save predictions to a JSON Lines file:
```bash
python predict.py --model_dir model_output \
                  --test_file data/NER-TESTING.jsonlines \
                  --output_file predictions.jsonlines
```

### 5. Postprocessing and Validation
Validate and correct the predictions:
```bash
python postprocess.py --input_file predictions.jsonlines \
                      --output_file corrected_predictions.jsonlines
```

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



