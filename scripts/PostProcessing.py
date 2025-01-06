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
from collections import defaultdict, Counter

#=======================================================================
#================================FUNCTION===============================
#=======================================================================
def resolve_token_tags(file_path, output_path):
    """
    Verifies that tokens have consistent NER tags and replaces them with the most frequent one.
    """
    token_tags = defaultdict(list)  # Dictionary to store all tags for each token
    valid_token_tags = {}  # To store the resolved valid tag for each token

    # First pass: Collect all tags for each token
    with open(file_path, 'r', encoding='utf-8') as file:  # Ensure UTF-8 encoding
        for line in file:
            data = json.loads(line)
            tokens = data.get("tokens", [])
            ner_tags = data.get("ner_tags", [])

            # Check if the number of tokens and ner_tags match
            if len(tokens) != len(ner_tags):
                print(f"Skipping entry with mismatched tokens and ner_tags lengths: {data['unique_id']}")
                continue

            # Track all tags for each token
            for token, tag in zip(tokens, ner_tags):
                token_tags[token].append(tag)

    # Second pass: Determine the most frequent tag for each token
    for token, tags in token_tags.items():
        tag_counts = Counter(tags)
        if len(tag_counts) > 1:
            # If there are multiple tags, resolve to the most frequent one
            most_common_tag, count = tag_counts.most_common(1)[0]
            valid_token_tags[token] = most_common_tag
            print(f"Inconsistency found for token '{token}': Replaced tags with '{most_common_tag}' based on frequency.")
        else:
            # If there's only one tag, consider it valid
            valid_token_tags[token] = tags[0]

    # Third pass: Replace the NER tags in the dataset with the resolved valid ones
    with open(file_path, 'r', encoding='utf-8') as file, open(output_path, 'w', encoding='utf-8') as output_file:
        for line in file:
            data = json.loads(line)
            tokens = data.get("tokens", [])
            ner_tags = data.get("ner_tags", [])

            # Replace the ner_tags with the resolved ones
            new_ner_tags = [valid_token_tags[token] for token in tokens]

            # Update the data with the new ner_tags and write it to the output file
            data['ner_tags'] = new_ner_tags
            json.dump(data, output_file, ensure_ascii=False)  # Write with non-ASCII characters preserved
            output_file.write('\n')  # Write each JSON object on a new line

    print("Token tags have been resolved and replaced.")
#-----------------------------------------------------------------------

def verify_token_tag_consistency(file_path):
    """
    Verifies that all the same tokens have the same NER tag across the JSON lines file.
    """
    token_tag_dict = {}  # Dictionary to store token-tag mappings
    inconsistency_count = 0

    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            tokens = data.get("tokens", [])
            ner_tags = data.get("ner_tags", [])

            # Check if the number of tokens and ner_tags match
            if len(tokens) != len(ner_tags):
                print(f"Skipping entry with mismatched tokens and ner_tags lengths: {data['unique_id']}")
                continue

            # Check token-tag consistency
            for token, tag in zip(tokens, ner_tags):
                if token not in token_tag_dict:
                    token_tag_dict[token] = tag
                elif token_tag_dict[token] != tag:
                    print(f"Inconsistency found for token '{token}' with tags {token_tag_dict[token]} and {tag}.")
                    inconsistency_count += 1

    print("Token consistency check completed.")
    print(f"Number of inconsistencies: {inconsistency_count}")
#-----------------------------------------------------------------------

def ensure_b_precedes_i(input_file, output_file):
    """
    Ensures that all `I-` tags in the dataset are preceded by the correct `B-` tag.
    
    Args:
        input_file (str): Path to the input JSON lines file.
        output_file (str): Path to the output corrected JSON lines file.

    Returns:
        int: Number of modifications made.
    """
    modifications = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(f_in, start=1): 
            data = json.loads(line.strip())
            tokens = data['tokens']
            ner_tags = data['ner_tags']
            original_tags = ner_tags.copy()  # Save original state for comparison

            for i in range(len(ner_tags)):
                if ner_tags[i].startswith('I-'):  
                    current_entity = ner_tags[i].split('-')[1]  
                    
                    if i == 0 or not ner_tags[i - 1].endswith(current_entity) or not ner_tags[i - 1].startswith('B-'):
                        # If the previous tag is not the correct `B-`, correct it
                        ner_tags[i] = 'B-' + current_entity
                        modifications += 1 

            if original_tags != ner_tags:
                print(f"Line {line_num}: Modifications detected")
                print(f"Before : {original_tags}")
                print(f"After : {ner_tags}\n")

            data['ner_tags'] = ner_tags
            # Write the data with proper encoding, ensuring special Unicode characters are preserved
            json.dump(data, f_out, ensure_ascii=False)  # This prevents escaping characters like `•` into unicode
            f_out.write('\n')  # Write each JSON object on a new line
    
    return modifications
#-----------------------------------------------------------------------

def process_ner_tags(input_file, output_file):
    """
    Replaces consecutive `B-` tags of the same entity with `I-`, except for the first occurrence.
    Also replaces special characters like bullets and arrows with their Unicode escape sequences.

    Args:
        input_file (str): Path to the input JSON lines file.
        output_file (str): Path to the output corrected JSON lines file.

    Returns:
        int: Number of modifications made.
    """
    modifications = 0
    """special_char_map = {
        '•': "\u2022",     # Bullet point
        '➔': '\\u2794',     # Rightwards arrow
        '↓': '\\u2193',     # Downwards arrow
        '–': '\\u2013',     # En Dash
        '«': '\\u00AB',     # Left-pointing double angle quotation mark
        '»': '\\u00BB',     # Right-pointing double angle quotation mark
        '⁰': '\\u2070',     # Superscript zero (if needed)
        '\uf0b7': '\\uF0B7', # Bullet character (Unicode variant)
        '§': '\\u00A7',     # Section symbol
        '\xad': '\\u00AD',   # Soft hyphen
        '少江': '\\u5C1A\\u6C5F',  # Chinese characters (example)
        '慧中': '\\u6085\\u4E2D',  # Chinese characters (example)
        '大学时代': '\\u5927\\u5B66\\u65F6\\u4EE3',  # Chinese characters (example)
        'Trojan\xadDownloader': '\\u0054\\u0072\\u006F\\u006A\\u0061\\u006E\\u00AD\\u0044\\u006F\\u0077\\u006E\\u006C0065\\u0072',  # Unicode of Trojan with soft hyphen included
        'User\xadAgent': '\\u0055\\u0073\\u0065\\u0072\\u00AD\\u0041\\u0067\\u0065\\u006E\\u0074',
        # Add more as needed...
    }"""

    with open(input_file, 'r', encoding='utf-8') as f_in, open(output_file, 'w', encoding='utf-8') as f_out:
        for line_num, line in enumerate(f_in, start=1):  
            data = json.loads(line)
            ner_tags = data["ner_tags"]  
            original_tags = ner_tags.copy()  # Save original state for comparison

            # Replace consecutive `B-` tags of the same entity with `I-`, except for the first occurrence
            for i in range(1, len(ner_tags)): 
                current_tag = ner_tags[i]
                previous_tag = ner_tags[i - 1]
                
                if current_tag.startswith('B-'):
                    current_entity = current_tag.split('-', 1)[1]
                    if previous_tag.endswith(current_entity):  
                        # If the current `B-` tag follows another tag of the same entity, replace it with `I-`
                        ner_tags[i] = current_tag.replace('B-', 'I-')
                        modifications += 1 
            
            # Replace special characters in tokens
            tokens = data["tokens"]
            for i in range(len(tokens)):
                token = tokens[i]
                # Replace special characters using the map
                """for char, escape in special_char_map.items():
                    token = token.replace(char, escape)"""
                tokens[i] = token  # Update token with the replaced special characters

            if original_tags != ner_tags:  
                print(f"Line {line_num}: Modifications detected")
                print(f"Before : {original_tags}")
                print(f"After : {ner_tags}\n")
            
            # Update the data with the new tokens and ner_tags
            data["tokens"] = tokens
            data["ner_tags"] = ner_tags
            
            # Write the data with proper encoding, ensuring special Unicode characters are preserved
            json.dump(data, f_out, ensure_ascii=True)
            f_out.write('\n')  # Write each JSON object on a new line
    
    return modifications

#-----------------------------------------------------------------------

#=======================================================================
#=================================MAIN==================================
#======================================================================= 
# Input and output file paths. To modify if needed
input_file = "predictions_output/predictions_output_1.jsonlines"
resolved_file = "resolved_file.jsonlines" # Resolved file / Do not change
ensure_b_file = "output_corrected.jsonlines"  # Transition file / Do not change
final_output_file = "corrected_predictions_output/corrected_predictions_output_1.jsonlines"  # Final output file

# Apply all the functions sequentially
# As we don't know how correct the NER tags (in our case, we based it on frequency), it might give worse results.
# So if wanted, you can turn into comments the 2 lines below and directly use the input file too apply NER rules corrections.
#resolve_token_tags(input_file, resolved_file)  # Resolve token tags
#verify_token_tag_consistency(resolved_file)  # Verify consistency after resolution

# Apply further NER rules corrections
print("Modification :", ensure_b_precedes_i(input_file, ensure_b_file))
#print("Modification :", ensure_b_precedes_i(resolved_file, ensure_b_file))
print("Modification :", process_ner_tags(ensure_b_file, final_output_file))

print("NER tag post-processing completed and merged.")