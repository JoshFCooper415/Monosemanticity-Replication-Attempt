import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from data_utils import process_batch

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, tied_weights=True):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim, dtype=torch.float32)
        if tied_weights:
            self.decoder = nn.Linear(hidden_dim, input_dim, dtype=torch.float32)
            self.decoder.weight = nn.Parameter(self.encoder.weight.t())
        else:
            self.decoder = nn.Linear(hidden_dim, input_dim, dtype=torch.float32)
        
        # Initialize weights with Xavier uniform
        nn.init.xavier_uniform_(self.encoder.weight)
        if not tied_weights:
            nn.init.xavier_uniform_(self.decoder.weight)
        
        # Initialize biases to small positive values
        nn.init.constant_(self.encoder.bias, 0.1)
        nn.init.constant_(self.decoder.bias, 0.1)

    def forward(self, x):
        encoded = F.relu(self.encoder(x))
        decoded = self.decoder(encoded)
        return decoded, encoded

def train_autoencoder(sae, dataloader, collector, device, num_epochs=10, l1_lambda=1e-5, learning_rate=1e-3):
    sae.to(device)
    sae = sae.float()
    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2)
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        sae.train()
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            activations = process_batch(collector, batch, device)
            
            optimizer.zero_grad()
            batch_loss = 0
            for act in activations:
                act = act.to(device).float()
                decoded, encoded = sae(act)
                
                reconstruction_loss = F.mse_loss(decoded, act)
                l1_loss = l1_lambda * encoded.abs().mean()
                
                loss = reconstruction_loss + l1_loss
                batch_loss += loss
            
            batch_loss /= len(activations)
            batch_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += batch_loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        
        # Print debugging information
        print("Debugging Information:")
        print(f"Encoder weight mean: {sae.encoder.weight.mean().item():.4f}, std: {sae.encoder.weight.std().item():.4f}")
        print(f"Encoder bias mean: {sae.encoder.bias.mean().item():.4f}, std: {sae.encoder.bias.std().item():.4f}")
        print(f"Decoder weight mean: {sae.decoder.weight.mean().item():.4f}, std: {sae.decoder.weight.std().item():.4f}")
        print(f"Decoder bias mean: {sae.decoder.bias.mean().item():.4f}, std: {sae.decoder.bias.std().item():.4f}")
        
        scheduler.step(avg_loss)
        
        # Validate and print activation statistics
        sae.eval()
        with torch.no_grad():
            val_loss, activation_mean, activation_std = validate_autoencoder(sae, dataloader, collector, device, l1_lambda)
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Activation Mean: {activation_mean:.4f}, Std: {activation_std:.4f}")
    
    return sae

def validate_autoencoder(sae, dataloader, collector, device, l1_lambda):
    total_loss = 0
    num_batches = 0
    all_activations = []
    for batch in dataloader:
        activations = process_batch(collector, batch, device)
        for act in activations:
            act = act.to(device).float()
            decoded, encoded = sae(act)
            reconstruction_loss = F.mse_loss(decoded, act)
            l1_loss = l1_lambda * encoded.abs().mean()
            loss = reconstruction_loss + l1_loss
            total_loss += loss.item()
            all_activations.append(encoded.cpu())
            num_batches += 1
        if num_batches >= 100:  # Limit validation to 100 batches
            break
    
    all_activations = torch.cat(all_activations, dim=0)
    return total_loss / num_batches, all_activations.mean().item(), all_activations.std().item()
