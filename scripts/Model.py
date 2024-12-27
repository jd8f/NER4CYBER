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
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from NERDataset import *
from DataProcessing import *

#=======================================================================
#================================FUNCTION===============================
#=======================================================================
# Model training
def train_model(model, train_loader, val_loader, loss_function, optimizer, epochs=3):
    """
    Train the model on the training data and validate after each epoch.
    
    Args:
        model: The model to be trained.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        loss_function: Loss function to optimize.
        optimizer: Optimizer for updating model parameters.
        epochs (int): Number of epochs to train.
    """
    model.train() # Set model to training mode
    for epoch in range(epochs):
        total_loss = 0
        correct_preds = 0
        total_preds = 0
        for batch in train_loader:
            # Move data to the appropriate device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad() # Clear previous gradients
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            # Compute weighted loss
            loss = loss_function(outputs.logits.view(-1, len(train_tag_to_idx)), labels.view(-1))
            loss.backward() # Backpropagation
            optimizer.step() # Update model parameters

            total_loss += loss.item()
            
            # Calculate accuracy
            preds = torch.argmax(outputs.logits, dim=-1) # Predicted class indices
            mask = (labels != tokenizer.pad_token_id)  # Ignore padding tokens
            correct_preds += (preds[mask] == labels[mask]).sum().item()
            total_preds += mask.sum().item()

        accuracy = correct_preds / total_preds
        print(f"-------------------------------------\n===============EPOCH={epoch + 1}===============\nAccuracy: {accuracy * 100:.2f}%\nLoss: {total_loss / len(train_loader)}")

        # Validate after each epoch
        validate_model(model, val_loader)
#-----------------------------------------------------------------------

def validate_model(model, val_loader):
    """
    Evaluate the model on validation data and calculate accuracy and loss.
    
    Args:
        model: The model to evaluate.
        val_loader: DataLoader for validation data.
    """
    model.eval() # Set model to evaluation mode
    total_loss = 0
    correct_preds = 0
    total_preds = 0
    #all_labels = []
    #all_preds = []
    
    with torch.no_grad(): # Disable gradient computation for validation
        for batch in val_loader:
            # Move data to the appropriate device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            # Compute the loss
            loss = loss_function(outputs.logits.view(-1, len(train_tag_to_idx)), labels.view(-1))
            total_loss += loss.item()
            
            # Calculate accuracy
            preds = torch.argmax(outputs.logits, dim=-1) # Predicted class indices
            mask = (labels != tokenizer.pad_token_id)  # Ignore padding tokens
            correct_preds += (preds[mask] == labels[mask]).sum().item()
            total_preds += mask.sum().item()
            
            # Retrieve predictions and ignore padding tokens
            #preds = torch.argmax(outputs.logits, dim=-1)
            #mask = (labels != tokenizer.pad_token_id)  # Mask to ignore padding tokens
            #all_labels.extend(labels[mask].cpu().numpy())
            #all_preds.extend(preds[mask].cpu().numpy())
            
    accuracy = correct_preds / total_preds
    print(f"Validation Accuracy: {accuracy * 100:.2f}%\nValidation Loss: {total_loss / len(val_loader)}")
    #print(classification_report(all_labels, all_preds, target_names=list(train_tag_to_idx.keys())))
    model.train() # Switch back to training mode
#-----------------------------------------------------------------------
  
#=======================================================================
#=================================MAIN==================================
#======================================================================= 
if __name__ == "__main__":
    
    # Convert class weights into an ordered list (aligned with tag_to_idx)
    weights_list = [train_class_weights[tag] for tag, idx in train_tag_to_idx.items()]
    weights_tensor = torch.tensor(weights_list, dtype=torch.float)

    #print("\nMapping des tags vers indices :", train_tag_to_idx)
    #print("Poids des classes :", weights_tensor)

    # Set the device to GPU if available, else use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_tensor = weights_tensor.to(device) # Transfer class weights to the device
    #print("\nPoids des classes :", weights_tensor)

    # Define the loss function with class weights
    loss_function = nn.CrossEntropyLoss(weight=weights_tensor)

    # Initialize the tokenizer and the mode
    model_name = "bert-base-uncased"  # Replace with the appropriate pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(train_tag_to_idx))
    
    # Prepare datasets
    train_dataset = NERDatasetTraining(preprocessed_train, tokenizer, train_tag_to_idx)
    val_dataset = NERDatasetTraining(preprocessed_val, tokenizer, train_tag_to_idx)

    # Prepare DataLoaders
    batch_size = 16  # Adjust batch size based on memory availability
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Move the model to GPU
    model = model.to(device)

    #print(f"Nombre de classes dans train_tag_to_idx : {len(train_tag_to_idx)}")

    # Start training the model
    print("\nStarting training...")
    train_model(model, train_loader, val_loader, loss_function, optimizer)

    # Save the trained model
    print("\nSaving model...")
    torch.save(model.state_dict(), "model/new_trained_ner_model_1.pth")
    print("MODEL SAVED")