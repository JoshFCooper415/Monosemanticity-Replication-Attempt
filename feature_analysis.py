import torch
from tqdm import tqdm
import logging
import os
from data_utils import process_batch
import numpy as np

def setup_logger(log_file):
    logger = logging.getLogger('feature_analysis')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def extract_correlated_sections(code, feature_activation, window_size=50, top_k=3):
    """
    Extract the most correlated sections of code for a given feature.
    
    :param code: The full code snippet
    :param feature_activation: Activation value for the feature for this snippet
    :param window_size: Size of the sliding window
    :param top_k: Number of top correlated sections to return
    :return: List of top correlated code sections
    """
    tokens = code.split()
    if len(tokens) < window_size:
        return [(' '.join(tokens[:min(len(tokens), 20)]), feature_activation)]  # Return at most 20 tokens

    windows = [' '.join(tokens[i:i+window_size]) for i in range(len(tokens) - window_size + 1)]
    
    # Since we only have one activation value for the entire snippet,
    # we'll return all windows with the same activation value
    return [(window, feature_activation) for window in windows[:top_k]]

def analyze_features(sae, dataloader, collector, device, model, tokenizer, log_dir='logs'):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'feature_analysis.log')
    logger = setup_logger(log_file)
    
    sae.to(device)
    sae.eval()

    feature_activations = [[] for _ in range(sae.encoder.out_features)]
    
    for batch in tqdm(dataloader, desc="Collecting feature activations"):
        activations = process_batch(collector, batch, device)
        _, _, codes = batch
        
        for act, code in zip(activations, codes):
            with torch.no_grad():
                act = act.to(device).to(torch.float32)
                _, encoded = sae(act)
            
            for i, feat_act in enumerate(encoded[0]):
                feature_activations[i].append((feat_act.item(), code))

    for feature in tqdm(range(len(feature_activations)), desc="Analyzing features"):
        logger.info(f"\n{'='*50}")
        logger.info(f"Analyzing Feature {feature}")
        logger.info(f"{'='*50}\n")

        # Sort activations for this feature
        sorted_activations = sorted(feature_activations[feature], key=lambda x: x[0], reverse=True)
        
        top_activations = sorted_activations[:5]  # Get top 5 activations

        logger.info("Top Activating Examples:")
        top_correlated_sections = []
        for i, (activation, snippet) in enumerate(top_activations):
            correlated_sections = extract_correlated_sections(snippet, activation)
            top_correlated_sections.extend(correlated_sections)
            logger.info(f"{i+1}. Most correlated code sections (activation: {activation:.4f}):")
            for j, (section, section_activation) in enumerate(correlated_sections):
                logger.info(f"   Section {j+1} (activation: {section_activation:.4f}): {section}\n")

        feature_label = label_feature_with_gemma(model, tokenizer, [section for section, _ in top_correlated_sections], device)
        logger.info(f"Gemma's interpretation of Feature {feature}:")
        logger.info(feature_label)
        logger.info("\n")

    print(f"Feature analysis complete. Results saved to {log_file}")

def label_feature_with_gemma(model, tokenizer, top_sections, device):
    prompt = f"""You are an AI assistant tasked with analyzing features of a sparse autoencoder trained on code snippets. Given the following information, please provide a concise label or description of what the feature might represent or detect in the code.

Top activating code sections:
{'-' * 40}
{chr(10).join(top_sections[:6])}  # Limiting to 6 sections to keep the prompt manageable
{'-' * 40}

Based on these code sections, what do you think this feature might be detecting or representing in the code? Please provide a concise label or description."""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_k=50,
            top_p=0.95,
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)