#_______________________________________________________________________
#Last name : FERNANDES
#First name : Jason Daryll
#Class : CDI
#Group : B
#Professor : M. KAAN ALKAN
#_______________________________________________________________________
#                               NER4CYBER
#_______________________________________________________________________

#=======================================================================
#=================================IMPORT================================
#=======================================================================
import json
from collections import Counter

#=======================================================================
#================================FUNCTION===============================
#=======================================================================
# Load the dataset
def load_dataset(filepath):
    """
    Load dataset from a JSON lines file.
    
    Args:
        filepath (str): Path to the JSON lines file.
    
    Returns:
        list: List of parsed JSON objects (samples).
    """
    with open(filepath, 'r') as file:
        return [json.loads(line) for line in file]
#-----------------------------------------------------------------------

# Load the Unicode tokens
def load_UNICODE(filepath):
    """
    Load dataset from a JSON lines file and identify tokens containing Unicode characters.
    
    Args:
        filepath (str): Path to the JSON lines file.
    
    Returns:
        tuple: List of dataset samples and a set of unique Unicode tokens.
    """
    unicode_tokens = set()
    
    with open(filepath, 'r') as file:
        dataset = []
        for line in file:
            example = json.loads(line)
            tokens = example.get("tokens", [])
            
            # Collect tokens with non-ASCII characters
            for token in tokens:
                if any(ord(c) > 127 for c in token):  # Check for non-ASCII characters
                    unicode_tokens.add(token)
            
            dataset.append(example)
    
    print(f"Found {len(unicode_tokens)} unique Unicode tokens.")
    return dataset, unicode_tokens
#-----------------------------------------------------------------------

# Analyze dataset
def analyze_dataset(data):
    """
    Analyze class distribution and token statistics for a dataset.
    
    Args:
        data (list): Dataset as a list of JSON objects.
    
    Returns:
        str: Summary of dataset statistics including tag distribution, token count, and vocabulary size.
    """
    ner_tags = [tag for sample in data for tag in sample['ner_tags']]
    tokens = [token for sample in data for token in sample['tokens']]
    
    # Distribution of NER tags
    tag_distribution = Counter(ner_tags)
    # Number of tokens
    token_count = len(tokens)
    vocab_size = len(set(tokens))
    
    return "\n---------------------------\ntag_distribution : " + str(tag_distribution) + "\n" + "token_count : " + str(token_count) + "\n" + "vocab_size : " + str(vocab_size) + "\n"
#-----------------------------------------------------------------------

# Preprocess data
def preprocess_data(data):
    """
    Prepare data for model input.
    Handles files with or without 'ner_tags' and ensures the presence of 'unique_id'.
    
    Args:
        data (list): Dataset as a list of JSON objects.
    
    Returns:
        list: Preprocessed data ready for model input.
    """
    processed_data = []
    for sample in data:
        # Extract existing values from the input file
        unique_id = sample.get("unique_id", None)  # Use the existing unique ID
        tokens = sample.get("tokens", [])
        ner_tags = sample.get("ner_tags", None)  # Check if 'ner_tags' exists
        
        # Optionally lowercase tokens
        #tokens = [token.lower() for token in tokens]
        
        if ner_tags is None:
            # If 'ner_tags' is missing, fill with a default value
            ner_tags = [-100] * len(tokens)  # -100 is ignored by loss functions in PyTorch
        
        # Ensure unique_id exists, or raise an error
        if unique_id is None:
            raise ValueError("Le champ 'unique_id' est manquant dans l'une des lignes.")
        
        processed_data.append({
            "unique_id": unique_id,  # Conserver l'ID unique du fichier original
            "tokens": tokens,
            "ner_tags": ner_tags
        })
        
    return processed_data
#-----------------------------------------------------------------------

# Calculate class weights
def calculate_class_weights(tag_distribution):
    """
    Calculate class weights inversely proportional to class frequencies.
    
    Args:
        tag_distribution (Counter): Distribution of NER tags.
    
    Returns:
        dict: Dictionary of class weights for each tag.
    """
    total = sum(tag_distribution.values())
    class_weights = {tag: (total / (len(tag_distribution) * count)) for tag, count in tag_distribution.items()}
    #class_weights = {tag: (total / count) for tag, count in tag_distribution.items()}

    class_weights['O'] = 0.1
    class_weights['I-Action'] = 0.75
    class_weights['I-Entity'] = 0.25
    class_weights['B-Action'] = 0.5
    class_weights['B-Modifier'] = 1.25
    class_weights['B-Entity'] = 0.5
    # Optional: Cap weights to avoid excessively large values
    max_weight_limit = 6  # or 80 depending on preference
    class_weights = {tag: min(weight, max_weight_limit) for tag, weight in class_weights.items()}

    return class_weights
#-----------------------------------------------------------------------
    
# Filepaths (update with actual file locations)
train_file = "data/NER-TRAINING.jsonlines"
val_file = "data/NER-VALIDATION.jsonlines"
test_file = "data/NER-TESTING.jsonlines"

# Load datasets
train_data = load_dataset(train_file)
val_data = load_dataset(val_file)
test_data = load_dataset(test_file)

#print("Training data : ",train_data,"\n")
#print("Validation data : ",val_data,"\n")
#print("Validation data : ",test_data)

# Load Unicode tokens
#_, test_unicode = load_UNICODE(test_file)
#print(test_unicode)

# Extract unique NER tags
train_tags = set(tag for sample in train_data for tag in sample["ner_tags"])

# Mapping tag -> index
train_tag_to_idx = {tag: idx for idx, tag in enumerate(sorted(train_tags))}
#print("Mapping des tags vers indices :", train_tag_to_idx)

# Analyze datasets
train_analysis = analyze_dataset(train_data)
val_analysis = analyze_dataset(val_data)

# Preprocess data
preprocessed_train = preprocess_data(train_data)
preprocessed_val = preprocess_data(val_data)
preprocessed_test = preprocess_data(test_data)

#print("Sample Preprocessed Data:", preprocessed_train[0])
#print("Sample Preprocessed Data:", preprocessed_test[44])
#print(len(preprocessed_test))

# Extract NER tag distributions
train_tag_distribution = Counter([tag for sample in train_data for tag in sample['ner_tags']])
val_tag_distribution = Counter([tag for sample in val_data for tag in sample['ner_tags']])

# Calculate class weights
train_class_weights = calculate_class_weights(train_tag_distribution) 
val_class_weights = calculate_class_weights(val_tag_distribution)

#=======================================================================
#=================================MAIN==================================
#=======================================================================
if __name__ == "__main__":
    
    print("=======================================================================\n")
    print("Training Data Analysis :", train_analysis)
    print("Validation Data Analysis :", val_analysis)
    print("=======================================================================\n")

    # Display the class weights
    print("Class Weights for Training Data\n--------------------------------")
    for tag, weight in train_class_weights.items():
        print(f"Tag: {tag}\t| Weight: {weight:.4f}")

    """print("\nClass Weights for Validation Data:")
    for tag, weight in val_class_weights.items():
        print(f"Tag: {tag}\t| Weight: {weight:.4f}")"""
