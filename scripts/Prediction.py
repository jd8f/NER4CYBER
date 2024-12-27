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
import torch
import numpy
import jsonlines 
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from NERDataset import *
from DataProcessing import *

#=======================================================================
#================================FUNCTION===============================
#=======================================================================
def predict_model(model, test_loader, idx_to_tag, tokenizer, output_file, unicode_replacements):
    """
    Predict NER tags for the test dataset and save results to a JSON lines file.

    Args:
        model: The trained NER model.
        test_loader: DataLoader for the test dataset.
        idx_to_tag: Dictionary mapping indices to tag names.
        tokenizer: Pre-trained tokenizer for input tokenization.
        output_file: Path to save prediction results.
        unicode_replacements: Dictionary for restoring original Unicode tokens.
    """
    model.eval() # Set the model to evaluation mode
    with torch.no_grad():
        with jsonlines.open(output_file, mode='w') as writer:
            for batch in test_loader:
                # Move data to the appropriate device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                unique_ids = batch["unique_id"]
                original_tokens_batch = batch["original_tokens"]

                # Get model predictions
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)

                for i in range(len(input_ids)):
                    # Get non-padding tokens
                    mask = (input_ids[i] != tokenizer.pad_token_id).cpu()
                    preds_ = preds[i][mask].cpu().numpy()
                    input_ids_ = input_ids[i][mask].cpu().numpy()
                    sub_tokens = tokenizer.convert_ids_to_tokens(input_ids_)

                    # Original tokens from batch
                    original_tokens = original_tokens_batch[i]

                    # Capitalize the first word of the tokens
                    #if len(original_tokens) > 0:
                    #   original_tokens[0] = original_tokens[0].capitalize()
                        
                    # Initialize NER tags list
                    ner_tags = []
                    token_idx = 0

                    for sub_token, pred_idx in zip(sub_tokens, preds_):
                        if sub_token in ["[CLS]", "[SEP]"]:
                            continue # Skip special tokens
                        elif sub_token.startswith("##"):
                            ner_tags[-1] = ner_tags[-1] # Extend the tag for the previous token
                        else:
                            # Assign a new NER tag for the original token
                            ner_tags.append(idx_to_tag[pred_idx])
                            token_idx += 1
                            if token_idx >= len(original_tokens):
                                break

                    # Restore orginal tokens for [UNK]
                    #for idx, token in enumerate(original_tokens):
                    #    if token == "[UNK]":
                    #        original_tokens[idx] = sub_tokens[idx]  # Replace [UNK] by original token

                    # Debugging output
                    #print(f"Original Tokens: {original_tokens}")
                    #print(f"Predicted NER Tags: {ner_tags}")
                    #print(f"Sub Tokens: {sub_tokens}")
                    #print(f"Predictions: {preds_}")

                    # Restore original tokens using Unicode replacement
                    restored_tokens = restore_unicode_tokens(original_tokens, unicode_replacements)

                    # Verify if the lengths match
                    if len(original_tokens) != len(ner_tags):
                        print(f"Length mismatch! Tokens: {len(original_tokens)} Tags: {len(ner_tags)}")
                        print()
                        # If mismatch, inspect more deeply here
                        continue

                    # Write prediction results to the output file      
                    writer.write({
                        "unique_id": unique_ids[i],
                        "tokens": restored_tokens,
                        "ner_tags": ner_tags
                    })

    print(f"Predictions saved to {output_file}.")
#-----------------------------------------------------------------------

def main():
    """
    Load the trained model and perform predictions on the test dataset.
    """
    # Initialize tokenizer and model
    model_name = "bert-base-uncased"  # Replace with your model name if different
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(train_tag_to_idx))

    # Load trained model weights
    model.load_state_dict(torch.load("model/new_trained_ner_model.pth", weights_only=True))
    model = model.to(device)

    # Create idx_to_tag mapping
    idx_to_tag = {idx: tag for tag, idx in train_tag_to_idx.items()}

    # Prepare test dataset and DataLoader
    test_dataset = NERDatasetPredict(preprocessed_test, tokenizer, train_tag_to_idx)
    unicode_replacements = test_dataset.unicode_replacements
    test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn)

    # Output file for predictions
    output_file = "predictions_output/predictions_output_1.jsonlines" # Change if needed

    # Perform predictions
    print("\nPrediction on the test dataset...")
    predict_model(model, test_loader, idx_to_tag, tokenizer, output_file, unicode_replacements)
#-----------------------------------------------------------------------

#=======================================================================
#=================================MAIN==================================
#======================================================================= 
if __name__ == "__main__":
    # Set the device to GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
