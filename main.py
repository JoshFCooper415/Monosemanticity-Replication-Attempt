import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

from model_wrapper import ActivationCollector
from sparse_autoencoder import SparseAutoencoder, train_autoencoder
from feature_analysis import analyze_features
from data_utils import create_dataloader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "google/gemma-2b-it"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    model.eval()

    collector = ActivationCollector(model, layer_num=model.config.num_hidden_layers // 2)

    max_length = 512
    batch_size = 16
    max_samples = 5000  # Limit to 1000 samples
    dataloader = create_dataloader("Joshfcooper/formai-v2-full-split-v2", split="train", tokenizer=tokenizer, 
                                   max_length=max_length, batch_size=batch_size, max_samples=max_samples)

    input_dim = model.config.hidden_size
    hidden_dim = input_dim * 2  # Reduced hidden dimension
    sae = SparseAutoencoder(input_dim, hidden_dim, tied_weights=True)

    sae = train_autoencoder(sae, dataloader, collector, device, num_epochs=1, l1_lambda=1e-5, learning_rate=1e-4)

    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Reset the dataloader for feature analysis
    dataloader = create_dataloader("Joshfcooper/formai-v2-full-split-v2", split="train", tokenizer=tokenizer, 
                                   max_length=max_length, batch_size=batch_size, max_samples=max_samples)

    analyze_features(sae, dataloader, collector, device, model, tokenizer, log_dir=log_dir)

    print("Analysis complete. Check the logs directory for results.")

if __name__ == "__main__":
    main()