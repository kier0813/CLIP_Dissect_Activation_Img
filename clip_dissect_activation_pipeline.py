import utils
import os
import math
import numpy as np
import pandas as pd
import torch
import clip
from tqdm import tqdm
from torch.utils.data import DataLoader
import data_utils

# Pipeline to run clip dissect analysis with activation images
def run_clip_dissect_analysis(layers, settings, d_probe, concept_set, similarity_fn):
    num_neurons_per_layer = [256, 512, 1024, 2048]
    data_list = []

    for layer_index, layer in enumerate(layers):
        # Initialize layer settings
        target_name = settings[layer]["target_name"]
        target_layer = settings[layer]["target_layer"]
        neurons_to_display = settings[layer]["neurons_to_display"]
        clip_name = 'ViT-B/16'
        d_probe = d_probe
        concept_set = concept_set
        batch_size = 50
        device = 'cuda'
        pool_mode = 'avg'
        save_dir = 'saved_activations'
        similarity_fn = similarity_fn

        # Calculate similarities
        utils.save_activations(clip_name=clip_name, target_name=target_name, target_layers=[target_layer],
                            d_probe=d_probe, concept_set=concept_set, batch_size=batch_size,
                            device=device, pool_mode=pool_mode, save_dir=save_dir)
        layer_broden = utils.get_save_names(clip_name=clip_name, target_name=target_name,
                                            target_layer=target_layer, d_probe=d_probe,
                                            concept_set=concept_set, pool_mode=pool_mode,
                                            save_dir=save_dir)
        target_layer_broden, clip_layer_broden, text_layer_broden = layer_broden
        similarities, target_feats = utils.get_similarity_from_activations(target_layer_broden, clip_layer_broden,text_layer_broden, similarity_fn, device=device)
        similarities_shape = similarities.shape
        
        if similarities_shape[0] != num_neurons_per_layer[layer_index]:
            raise ValueError(f"Unexpected number of neurons in layer {layer}. Expected: {num_neurons_per_layer[layer_index]}, Found: {similarities_shape[0]}")

        with open(concept_set, 'r') as f:
            words = [line.strip() for line in f if line.strip()]  

        if not words:
            raise ValueError("The 'words' list is empty. Check the 'concept_set' file contents and path.")

        # Get data for the probe
        pil_data = data_utils.get_data(d_probe)   
        num_neurons = similarities.shape[0]
        for neuron_id in range(num_neurons_per_layer[layer_index]):
            # Get the top 5, 10, and 16 activation indices
            _, indices_5 = torch.topk(similarities[neuron_id, :], k=5, largest=True)
            _, indices_10 = torch.topk(similarities[neuron_id, :], k=10, largest=True)
            _, indices_16 = torch.topk(similarities[neuron_id, :], k=16, largest=True)

            # Convert indices to labels
            labels_5 = [words[idx] for idx in indices_5]
            labels_10 = [words[idx] for idx in indices_10[:5]]  # Only consider the top 5
            labels_16 = [words[idx] for idx in indices_16[:5]]  # Only consider the top 5

            # Check if the sets of labels are the same (which implies rank stability)
            stability_5_10 = set(labels_5) == set(labels_10)
            stability_5_16 = set(labels_5) == set(labels_16)

            new_row = {
                'Layer': layer,
                'Neuron ID': neuron_id,
                'Labels k=5': labels_5,
                'Labels k=10': labels_10,
                'Labels k=16': labels_16,
                'Stability 5-10': stability_5_10,
                'Stability 5-16': stability_5_16
            }
            data_list.append(new_row)
            
    results_df = pd.DataFrame(data_list)
    
    # Calculate stability
    stability_score_5_10 = results_df['Stability 5-10'].sum() / len(results_df) * 100
    stability_score_5_16 = results_df['Stability 5-16'].sum() / len(results_df) * 100

    # Print the stability scores
    print(f"Stability Score from k=5 to k=10: {stability_score_5_10}%")
    print(f"Stability Score from k=5 to k=16: {stability_score_5_16}%")

    return results_df, stability_score_5_10, stability_score_5_16