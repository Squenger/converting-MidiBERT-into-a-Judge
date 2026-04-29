import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from data_generation.reward_dataset import MidiBERTRewardDataset
from miditok import REMI
import time
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f" CUDA device: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# initialization (regression mode)
# this tells the transformer to end with a single neuron (our score out of 10).
print("loading midibert on the gpu...")
judge_model = AutoModelForSequenceClassification.from_pretrained(
    "manoskary/musicbert",
    num_labels=1     
).to(device)


path_to_data = "/Users/aiminemeddeb/Documents/Music_Generation_Transformers/data/maestro-v3.0.0"


tokenizer = REMI()

data = torch.load("./data_generation/maestro_tokenized_complet.pt")

# split data into training and validation (90/10)
split_idx = int(0.9 * len(data))
train_data = data[:split_idx]
val_data = data[split_idx:]

train_dataset = MidiBERTRewardDataset(train_data, tokenizer)
val_dataset = MidiBERTRewardDataset(val_data, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# we use a small learning rate because midibert already knows music
optimizer = AdamW(judge_model.parameters(), lr=2e-5)

# loss function: mse
loss_fn = nn.MSELoss() 

epochs = 50 

# training loop
print("starting training")
start_time = time.time()
all_train_losses = []
all_val_losses = []
best_val_loss = float("inf")

for epoch in range(epochs):
    cumulative_train_loss = 0.0
    judge_model.train()
    
    for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device) 
        
        optimizer.zero_grad()
        
        # forward pass 
        outputs = judge_model(batch_x)
        predictions = outputs.logits 
        
        # error
        loss = loss_fn(predictions, batch_y)
        
        # backward pass 
        loss.backward()
        optimizer.step()
        
        cumulative_train_loss += loss.item()
        all_train_losses.append(loss.item())
        
        # display every 50 iterations
        if batch_idx % 50 == 0:
            print(f"epoch {epoch+1}/{epochs} | batch {batch_idx} | train mse loss: {loss.item():.4f}")

    # validation loop
    judge_model.eval()
    cumulative_val_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in val_dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = judge_model(batch_x)
            loss = loss_fn(outputs.logits, batch_y)
            cumulative_val_loss += loss.item()
    
    avg_val_loss = cumulative_val_loss / len(val_dataloader)
    all_val_losses.append(avg_val_loss)
    print(f"epoch {epoch+1}/{epochs} | validation mse loss: {avg_val_loss:.4f}")
    
    # save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        judge_model.save_pretrained("./midibert_judge_best")
        print(f"new best model saved with val loss: {avg_val_loss:.4f}")

print("training completed")
end_time = time.time()
print(f"total training time: {end_time - start_time:.2f} seconds")

# plot loss curve
plt.figure(figsize=(10, 5))
plt.plot(all_train_losses, label="train loss")
# scaling val loss to match step count for visualization
val_steps = len(train_dataloader)
plt.plot([i * val_steps for i in range(1, epochs + 1)], all_val_losses, label="val loss", marker='o')
plt.title("training and validation loss")
plt.xlabel("steps")
plt.ylabel("mse loss")
plt.legend()
plt.savefig("loss_curve.png")


judge_model.save_pretrained("./midibert_judge")
print("saved")
