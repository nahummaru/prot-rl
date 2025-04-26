import torch
from transformers import AutoTokenizer, EsmForProteinFolding
import numpy as np

def compute_stability(sequence):
    """
    Compute protein stability score using ESMFold.
    Returns a normalized score between 0 and 1 based on:
    1. pLDDT scores (confidence in structure prediction)
    2. PAE (predicted aligned error) scores
    3. Structure compactness
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load ESMFold
    tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1").to(device)
    model.eval()
    
    try:
        with torch.no_grad():
            # Get structure prediction and confidence scores
            output = model.infer_pdb(sequence)
            
            # Extract pLDDT scores (per-residue confidence)
            plddt_scores = output.plddt  # Shape: [L], where L is sequence length
            mean_plddt = plddt_scores.mean().item() / 100.0  # Normalize to 0-1
            
            # Extract PAE scores (predicted aligned error)
            pae = output.predicted_aligned_error  # Shape: [L, L]
            mean_pae = pae.mean().item()
            normalized_pae = np.exp(-mean_pae / 10.0)  # Convert to 0-1 score, lower PAE is better
            
            # Calculate structure compactness
            coords = output.positions  # Shape: [L, 14, 3] for backbone + CB atoms
            ca_coords = coords[:, 1]  # Alpha carbon coordinates
            dists = torch.cdist(ca_coords, ca_coords)
            radius_of_gyration = torch.sqrt(torch.mean(torch.square(dists))).item()
            compactness = np.exp(-radius_of_gyration / 30.0)  # Normalize to 0-1
            
            # Combine scores with weights
            stability_score = (
                0.4 * mean_plddt +  # Structure confidence
                0.4 * normalized_pae +  # Structure quality
                0.2 * compactness  # Structure compactness
            )
            
            return stability_score
            
    except Exception as e:
        print(f"Error computing stability: {e}")
        return 0.0
    finally:
        # Clean up
        del model
        torch.cuda.empty_cache() 