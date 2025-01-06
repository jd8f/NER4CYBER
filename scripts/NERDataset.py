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
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence

#=======================================================================
#=================================CLASS=================================
#=======================================================================
class NERDatasetTraining(Dataset):
    """
    Dataset class for training NER models.
    """
    def __init__(self, data, tokenizer, tag_to_idx, max_len=256):
        """
        Args:
            data (list of dict): Input data with tokens and labels.
            tokenizer (AutoTokenizer): Pre-trained tokenizer.
            tag_to_idx (dict): Mapping of tags to their respective indices.
            max_len (int): Maximum input sequence length (default: 256).
        """
        self.data = data
        self.tokenizer = tokenizer
        self.tag_to_idx = tag_to_idx
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        tokens = example["tokens"]
        
        # Remplace Unicode characters
        tokens = [token if not any(ord(c) > 127 for c in token) else "[UNK]" for token in tokens]

        # Include unique_id in the dataset
        unique_id = example.get("unique_id", idx)  # Use `idx` as a fallback if no unique_id is present

        # Tokenize input text with padding and truncation
        encoding = self.tokenizer(
            tokens,
            truncation=True,  # Ensure truncation to max_len
            padding='max_length',  # Pad to max_len
            max_length=self.max_len,
            is_split_into_words=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)  # Shape: [max_len]
        attention_mask = encoding['attention_mask'].squeeze(0)  # Shape: [max_len]

        # Ensure the padding is correctly handled
        assert input_ids.size(0) == self.max_len, f"Input ids length mismatch: {input_ids.size(0)} != {self.max_len}"
        assert attention_mask.size(0) == self.max_len, f"Attention mask length mismatch: {attention_mask.size(0)} != {self.max_len}"

        # Get labels (ensure they're aligned with input_ids)
        ner_tags = example["ner_tags"]
        labels = [self.tag_to_idx.get(tag, self.tag_to_idx["O"]) for tag in ner_tags]

        # Padding labels to max_len (to match input_ids length)
        padding_length = self.max_len - len(labels)
        labels += [self.tag_to_idx["O"]] * padding_length  # Padding labels with "O"
        
        # Ensure labels have correct size
        labels = labels[:self.max_len]  # Truncate if necessary
        labels = torch.tensor(labels)  # Convert labels to tensor

        # Debug: Check the size of labels
        assert labels.size(0) == self.max_len, f"Labels length mismatch: {labels.size(0)} != {self.max_len}"

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'unique_id': unique_id,  # Add unique_id here
        }
#-----------------------------------------------------------------------

class NERDatasetPredict(Dataset):
    """
    Dataset class for prediction (test) NER models.
    """
    def __init__(self, data, tokenizer, tag_to_idx, max_len=256):
        """
        Args:
            data (list of dict): Input data with tokens and optionally labels.
            tokenizer (AutoTokenizer): Pre-trained tokenizer.
            tag_to_idx (dict): Mapping of tags to indices.
            max_len (int): Maximum input sequence length.
            is_test (bool): Whether the data is for testing (no labels).
        """
        self.data = data
        self.tokenizer = tokenizer
        self.tag_to_idx = tag_to_idx
        self.max_len = max_len
        self.unicode_replacements = {}  # Dictionnary to stock Unicodes
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        tokens = example["tokens"]
        
        # Include unique_id in the dataset
        unique_id = example.get("unique_id", idx)  # Use `idx` as a fallback if no unique_id is present

        # Remplace Unicode characters by [UNK] and stock them 
        modified_tokens = []
        for i, token in enumerate(tokens):
            if any(ord(c) > 127 for c in token):  # If non-ASCII
                # Replace with the Unicode escape sequence of the token
                unicode_token = token.encode('unicode_escape').decode('utf-8')  # This will give '\u2022' without quotes
                unk_token = f"[UNK-{idx}-{i}]"  # Unique id for replacement
                self.unicode_replacements[unk_token] = unicode_token  # Save original
                print(self.unicode_replacements)  # Inspect all the unicode tokens
                modified_tokens.append(unk_token)
            else:
                modified_tokens.append(token)

        # Rest unchanged
        tokens = modified_tokens
        
        # Tokenize input text with padding and truncation
        encoding = self.tokenizer(
            tokens,
            truncation=True,  # Ensure truncation to max_len
            padding='max_length',  # Pad to max_len
            max_length=self.max_len,
            is_split_into_words=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)  # Shape: [max_len]
        attention_mask = encoding['attention_mask'].squeeze(0)  # Shape: [max_len]

        # Ensure the padding is correctly handled
        assert input_ids.size(0) == self.max_len, f"Input ids length mismatch: {input_ids.size(0)} != {self.max_len}"
        assert attention_mask.size(0) == self.max_len, f"Attention mask length mismatch: {attention_mask.size(0)} != {self.max_len}"

        # For test data, return input_ids and attention_mask (no labels)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'unique_id': unique_id,  # Include unique_id for test data as well
            'original_tokens': tokens,  # Include the original tokens
        }
#-----------------------------------------------------------------------

#=======================================================================
#================================FUNCTION===============================
#=======================================================================
def collate_fn(batch):
    """
    Custom collate function to handle batching for the DataLoader.
    """
    # Get the sequences and their corresponding labels
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    original_tokens = [item['original_tokens'] for item in batch]
    unique_ids = [item['unique_id'] for item in batch]

    # Pad the input_ids and attention_masks to the same length
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)  # Padding with 0 (usually the pad token ID)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    # Make sure all input data are in the correct tensor format
    input_ids_padded = input_ids_padded.to(torch.long)
    attention_masks_padded = attention_masks_padded.to(torch.long)

    # Return a dictionary with the padded sequences and the original data
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'unique_id': unique_ids,
        'original_tokens': original_tokens
    }
#-----------------------------------------------------------------------

def restore_unicode_tokens(tokens, unicode_replacements):
    """
    Restore the original Unicode tokens that were replaced during preprocessing.
    """
    return [unicode_replacements.get(token, token) for token in tokens]