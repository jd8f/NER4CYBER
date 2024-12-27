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
from DataProcessing import load_dataset

#=======================================================================
#================================FUNCTION===============================
#=======================================================================
def validate_ner_structure(tokens, predicted_tags):
    """
    Validate the structure of NER tags for a sequence.
    Checks transitions between tags, ensuring they follow the expected format:
    - 'I-' tags must follow 'B-' or another 'I-' tag of the same entity.
    - 'O' should not incorrectly split 'B-' and 'I-' tags.
    """
    valid = True
    errors = []
    
    # Iterate through predicted tags and check for structural errors
    for i in range(1, len(predicted_tags)):
        current_tag = predicted_tags[i]
        previous_tag = predicted_tags[i - 1]

        # Check invalid transitions for "I-" tags (must follow a "B-" or same "I-" tag)
        if current_tag.startswith("I-"):
            if not (previous_tag == f"B-{current_tag[2:]}" or previous_tag == current_tag):
                errors.append(
                    f"Invalid transition at token '{tokens[i]}' ({i}): {previous_tag} -> {current_tag}"
                )
                valid = False

        # Check if 'O' splits B- and I- tags incorrectly
        if current_tag == "O" and previous_tag.startswith("B-") and i + 1 < len(predicted_tags) and predicted_tags[i + 1].startswith("I-"):
            errors.append(
                f"Invalid position of 'O' at token '{tokens[i]}' ({i}): {previous_tag} -> O -> {predicted_tags[i + 1]}"
            )
            valid = False

    return valid, errors
#-----------------------------------------------------------------------

def extract_entities(tokens, predicted_tags):
    """
    Extract entities detected from tokens and predicted tags.
    Returns a list of tuples where each tuple contains an entity (string) and its corresponding label.
    """
    entities = []
    current_entity = []
    current_label = None

    # Iterate through the tokens and tags to identify entities
    for token, tag in zip(tokens, predicted_tags):
        if tag.startswith("B-"):
            # Save the previous entity
            if current_entity:
                entities.append((" ".join(current_entity), current_label))
            # Start a new entity
            current_entity = [token]
            current_label = tag[2:]
        elif tag.startswith("I-") and current_label == tag[2:]:
            # Continue adding tokens to the current entity if valid
            current_entity.append(token)
        else:
            # Save and reset if encountering a non-continuing token
            if current_entity:
                entities.append((" ".join(current_entity), current_label))
            current_entity = []
            current_label = None

    # Add the last entity if present
    if current_entity:
        entities.append((" ".join(current_entity), current_label))

    return entities
#-----------------------------------------------------------------------

def check_global_consistency(predictions):
    """
    Check global consistency of entity labels across all sequences.
    Identifies if the same entity is assigned different labels in different occurrences.
    """
    entity_labels = {}
    inconsistencies = []

    for example in predictions:
        tokens = example["tokens"]
        predicted_tags = example["ner_tags"]
        entities = extract_entities(tokens, predicted_tags)

        # Check for label inconsistencies across different occurrences of the same entity
        for entity, label in entities:
            if entity not in entity_labels:
                entity_labels[entity] = label
            elif entity_labels[entity] != label:
                inconsistencies.append(
                    f"Inconsistent label for entity '{entity}': "
                    f"{entity_labels[entity]} vs {label}"
                )

    return inconsistencies
#-----------------------------------------------------------------------

def main():
    # Define the output path for predictions.
    # Modify to check before and after post processing.
    input_path = "predictions_output/predictions_output_1.jsonlines"
    #input_path = "corrected_predictions_output/corrected_predictions_output_1.jsonlines"
    predictions = load_dataset(input_path)

    # Check for global inconsistencies across all predictions
    inconsistencies = check_global_consistency(predictions)
    print("Inconsistency cases :", len(inconsistencies))

    # Check for structural errors in each example
    structural_errors = 0
    for example in predictions:
        tokens = example["tokens"]
        predicted_tags = example["ner_tags"]
        valid, errors = validate_ner_structure(tokens, predicted_tags)
        if not valid:
            structural_errors += len(errors)

    print("Error cases:", structural_errors)
#-----------------------------------------------------------------------

#=======================================================================
#=================================MAIN==================================
#======================================================================= 
if __name__ == "__main__":
    main()