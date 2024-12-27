
# NER4CYBER: Named Entity Recognition for Cybersecurity

This repository contains a Named Entity Recognition (NER) project designed to identify and classify cybersecurity-related entities within text. The system is based on a fine-tuned BERT model and includes scripts for data preparation, model training, evaluation, and prediction, tailored to the cybersecurity domain.

## Features
- **Domain-Specific NER**: Identifies cybersecurity entities like attacks, threats, vulnerabilities, and other cybersecurity-related terms.
- **BERT-based Model**: Fine-tuned BERT model optimized for NER tasks in cybersecurity.
- **Class Imbalance Handling**: Utilizes weighted loss to address class imbalance during training.
- **IOB2 Tagging Support**: Enforces IOB2 tagging consistency for training and validation.
- **Utility Scripts**: Includes scripts for data preprocessing, model training, prediction generation, and postprocessing.

---

## Setup Instructions

### Prerequisites
1. **Python 3.8 or later**: Ensure that Python 3.8+ is installed on your system.
2. **Git**: Git must be installed to clone the repository.

### Installation
Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/ner-cybersecurity.git
   cd ner-cybersecurity
   ```

---

## Dataset Preparation

Ensure that your dataset is in JSON Lines format and placed in the `data/` directory. The dataset files should be structured as follows:

- **`NER-TRAINING.jsonlines`**: Contains the training data.
- **`NER-VALIDATION.jsonlines`**: Contains the validation data.
- **`NER-TESTING.jsonlines`**: Contains the testing data.

Each JSON object in the dataset should follow this structure:

```json
{
    "unique_id": <int>,         # A unique identifier for the tokenized example
    "tokens": ["token1", "token2", ...],  # Tokenized text
    "ner_tags": ["O", "B-Entity", "I-Entity", ...]  # Corresponding NER tags
}
```

Where:
- `O` denotes a token that is not part of any named entity.
- `B-Entity` and `I-Entity` represent the beginning and inside of a named entity, respectively.

---

## How to Run the Project

### 1. Preprocessing Data
Before training, preprocess your data for tokenization and alignment:
- Ensure that the paths to the training, validation, and testing datasets are correct in `DataProcessing.py`.
- Run the script to prepare your data for the model:
   ```bash
   python scripts/DataProcessing.py
   ```

### 2. Training the Model
Once your data is prepared, you can begin training the NER model:
- Make sure the new model path and the dataset paths are correct.
- Train the model by running:
   ```bash
   python scripts/Model.py
   ```

### 3. Generating Predictions on Test Data
After training the model, use it to generate predictions on the test dataset:
- Specify the model and output (output prediction) file paths.
- Run the prediction script to generate predictions:
   ```bash
   python scripts/Prediction.py
   ```

### 4. Postprocessing and Validation
Ensure the structure and consistency of the predictions before using them:
- Validate the output predictions using `ValidationStructureConsistency.py`:
   ```bash
   python scripts/ValidationStructureConsistency.py
   ```

- If any structural issues are found, ensure the input (output prediction) and output (corrected output prediction) paths are correct and run the postprocessing script to correct them:
   ```bash
   python scripts/PostProcessing.py
   ```

- After postprocessing, rerun the validation to confirm correctness:
   ```bash
   python scripts/ValidationStructureConsistency.py
   ```

---

## Repository Structure
```
.
├── corrected_predictions_output/             # Directory for corrected prediction output files.
├── data/                                     # Directory for raw datasets.
├── model/                                    # Directory for trained model outputs.
├── predictions_output/                       # Directory for predictions output files.
├── scripts/                                  # Directory containing utility scripts.
│   ├── DataProcessing.py                     # Script for data preprocessing.
│   ├── Model.py                              # Script for model training.
│   ├── NERDataset.py                         # Script for dataset evaluation.
│   ├── PostProcessing.py                     # Script for postprocessing corrections.
│   ├── Prediction.py                         # Script for generating predictions.
│   ├── ValidationStructureConsistency.py     # Script for validating prediction structure.
├── README.md                                 # Project documentation.
```

---

## Future Improvements
- **Model Performance**: Enhance the model's ability to handle rare tags like `I-Modifier` by incorporating techniques like data augmentation or transfer learning.
- **Alternative Models**: Explore alternative transformer architectures like RoBERTa, ELECTRA, or DistilBERT for potentially better performance.
- **Error Analysis**: Introduce detailed error analysis and provide insights on misclassifications for further refinement.
- **Nested Entities**: Add support for recognizing nested or overlapping entities to better handle complex entity relationships in text.

---
