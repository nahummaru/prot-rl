#!/usr/bin/env python3
import os
import sys
import tempfile
import logging
import subprocess
from typing import List, Tuple

import torch
import numpy as np
import esm
from esm.inverse_folding.util import (
    load_structure,
    extract_coords_from_structure,
    CoordBatchConverter,
)
from transformers import AutoTokenizer, EsmForProteinFolding

from Bio.PDB import PDBParser

import time 

torch.backends.cuda.matmul.allow_tf32 = True

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Globals & fit params
# -----------------------------------------------------------------------------
_IF_CHECKPOINT   = "esm_if1_gvp4_t16_142M_UR50.pt"
_IF_URL          = "https://sid.erda.dk/share_redirect/eIZVVNEd8B"
_FIT_A, _FIT_B   = 0.10413378327743603, 0.6162549378400894

_efs_tokenizer   = None
_efs_model       = None

_if_model        = None
_alphabet        = None
_batch_converter = None

# -----------------------------------------------------------------------------
# Your provided utilities
# -----------------------------------------------------------------------------

def masked_absolute(mut, idx, token_probs, alphabet):
    mt_encoded = alphabet.get_idx(mut)
    score = token_probs[0,idx, mt_encoded]
    return score.item()
     
def run_model(coords, sequence, model, cmplx=False, chain_target='A'):
    device = next(model.parameters()).device

    batch_converter = CoordBatchConverter(_alphabet)
    if isinstance(coords, list) and isinstance(sequence, list):
        print("=== running batched stability scoring ===")
        batch = [(c, None, s) for c, s in zip(coords, sequence)]
    else:
        batch = [(coords, None, sequence)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(
        batch, device=device)

    prev_output_tokens = tokens[:, :-1].to(device)
    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)

    logits_swapped=torch.swapaxes(logits,1,2)
    token_probs = torch.softmax(logits_swapped, dim=-1)

    return token_probs

def score_variants(sequence,token_probs,alphabet):
    aa_list=[]
    wt_scores=[]
    skip_pos=0

    alphabetAA_L_D={'-':0,'_' :0,'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20}
    alphabetAA_D_L={v: k for k, v in alphabetAA_L_D.items()}

    for i,n in enumerate(sequence):
      aa_list.append(n+str(i+1))
      score_pos=[]
      for j in range(1,21):
          score_pos.append(masked_absolute(alphabetAA_D_L[j],i, token_probs, alphabet))
          if n == alphabetAA_D_L[j]:
            WT_score_pos=score_pos[-1]

      wt_scores.append(WT_score_pos)

    return aa_list, wt_scores

def _score_variants(sequence, token_probs, alphabet):
    wt_scores = []
    aa_list = []
    skip_pos = 0
    alphabetAA_L_D = {
        '-': 0,  '_' : 0,
        'A':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,
        'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,
        'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20
    }
    alphabetAA_D_L = {v:k for k,v in alphabetAA_L_D.items()}


    for i,n in enumerate(sequence):
      aa_list.append(n+str(i+1))
      score_pos=[]
      for j in range(1,21):
          score_pos.append(masked_absolute(alphabetAA_D_L[j],i, token_probs, alphabet))
          if n == alphabetAA_D_L[j]:
            WT_score_pos=score_pos[-1]

      wt_scores.append(WT_score_pos)

    return aa_list, wt_scores


    for i, wt in enumerate(sequence):
        # compute probabilities for all 20, but pick wild-type
        j = alphabetAA_L_D[wt]
        score = token_probs[0, j, i].item()
        wt_scores.append(score)
    return wt_scores

# -----------------------------------------------------------------------------
# Ensure we have the ESM-IF checkpoint on disk
# -----------------------------------------------------------------------------
def _ensure_if_checkpoint():
    if os.path.isfile(_IF_CHECKPOINT):
        return
    logger.info(f"Downloading ESM-IF checkpoint to '{_IF_CHECKPOINT}' via wget…")
    subprocess.run([
        "wget", "--quiet",
        "-O", _IF_CHECKPOINT,
        _IF_URL
    ], check=True)

# -----------------------------------------------------------------------------
# ESMFold loader & folding
# -----------------------------------------------------------------------------
def _load_esmfold(model_name: str = "facebook/esmfold_v1"):
    global _efs_tokenizer, _efs_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _efs_model is None:
        logger.info(f"Loading ESMFold model ({model_name})…")
        # TODO (nahum): Seems to be in loading weights.  
        """ ome weights of EsmForProteinFolding were not initialized from the model checkpoint at facebook/esmfold_v1 and are newly initialized: ['esm.contact_head.regression.bias', 'esm.contact_head.regression.weight']
        You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."""
        _efs_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _efs_model = EsmForProteinFolding.from_pretrained(model_name)
        _efs_model.eval()
        _efs_model.esm = _efs_model.esm.half()
        _efs_model = _efs_model.to(device)
    return _efs_model, device

def _fold_seq_to_pdb_str(sequence: str, model_name: str) -> Tuple[str, float]:
    """Fold sequence and return both PDB string and mean pLDDT score."""
    model, _ = _load_esmfold(model_name)

    results = []
    mean_plddts = []

    with torch.no_grad():
        start = time.time()
        print("max len here", max([len(seq) for seq in sequence]))
        outputs = model.infer_pdbs(sequence)
        print(f"Time taken to fold single sequence: {time.time() - start} seconds")
        
        # Extract pLDDT scores from the PDB B-factor column
        for output in outputs:
            plddt_scores = []
            for line in output.split('\n'):
                if line.startswith('ATOM'):
                    try:
                        plddt = float(line[60:66].strip())  # B-factor column contains pLDDT
                        plddt_scores.append(plddt)
                    except (ValueError, IndexError):
                        continue
            
            mean_plddt = np.mean(plddt_scores) if plddt_scores else np.nan

            results.append(output)
            mean_plddts.append(mean_plddt)

    return results, mean_plddts

# -----------------------------------------------------------------------------
# ESM-IF loader & scoring
# -----------------------------------------------------------------------------
def _load_if_model():
    global _if_model, _alphabet, _batch_converter
    _ensure_if_checkpoint()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _if_model is None:
        logger.info(f"Loading ESM-IF model ({_IF_CHECKPOINT})…")
        # allow argparse.Namespace unpickling if needed
        from torch.serialization import safe_globals
        import argparse
        with safe_globals([argparse.Namespace]):
            _if_model, _alphabet = esm.pretrained.load_model_and_alphabet(_IF_CHECKPOINT)
        _if_model = _if_model.eval().to(device)
        _batch_converter = CoordBatchConverter(_alphabet)
    return _if_model, _alphabet, _batch_converter, device

def _pdb_to_coord_seq(pdb_str: str, chain_id: str):
    # write pdb to temp file
    tmp = tempfile.NamedTemporaryFile(mode="w+", suffix=".pdb", delete=False)
    tmp.write(pdb_str)
    tmp.flush()
    tmp.close()
    tmp_path = tmp.name

    try:
        struct = load_structure(tmp_path, chain_id)
        coords, seq = extract_coords_from_structure(struct)

    finally:
        os.remove(tmp_path)
    
    return coords, seq

def _pdb_to_coord_seq_batch(pdb_strs: List[str], chain_id: str):
    coords = []
    seqs = []

    for pdb_str in pdb_strs:
        coord, seq = _pdb_to_coord_seq(pdb_str, chain_id)
        coords.append(coord)
        seqs.append(seq)
        
    return coords, seqs

def _score_pdb_str_batch(pdb_strs: List[str], chain_id: str):
    coords, seqs = _pdb_to_coord_seq_batch(pdb_strs, chain_id)

    model, alphabet, _, device = _load_if_model()
    probs = run_model(coords, seqs, model)

    out = []

    for i in range(probs.shape[0]):
        aa_list, wt_scores = score_variants(seqs[i], probs[i].unsqueeze(0), alphabet)
        raw_if = float(np.sum(wt_scores))
        dg  = _FIT_A * raw_if + _FIT_B
        out.append((raw_if, dg))

    return out

def _score_pdb_str(pdb_str: str, chain_id: str) -> Tuple[float, float]:
    # write pdb to temp file
    tmp = tempfile.NamedTemporaryFile(mode="w+", suffix=".pdb", delete=False)
    tmp.write(pdb_str)
    tmp.flush()
    tmp.close()
    tmp_path = tmp.name

    try:
        model, alphabet, _, device = _load_if_model()
        struct = load_structure(tmp_path, chain_id)
        coords, seq = extract_coords_from_structure(struct)

        # use your provided helpers
        probs = run_model(coords, seq, model)
        aa_list, wt_scores = score_variants(seq, probs, alphabet)

        raw_if = float(np.sum(wt_scores))
        dg  = _FIT_A * raw_if + _FIT_B

    finally:
        os.remove(tmp_path)

    return raw_if, dg

# -----------------------------------------------------------------------------
# Public API: batch stability
# -----------------------------------------------------------------------------
def stability_score(
    sequences: List[str],
    chain_id: str = "A",
    esmfold_model: str = "facebook/esmfold_v1"
) -> List[Tuple[float, float, float]]:  # Updated return type to include pLDDT
    """
    Batch absolute ΔG prediction:
      1) fold each seq with ESMFold
      2) score each structure with ESM-IF

    Returns list of (raw_IF_score, ΔG_kcal/mol, pLDDT).
    """
    _load_esmfold(esmfold_model)
    _load_if_model()

    results = []
    for seq in sequences:
        if not seq or any(aa not in "ACDEFGHIKLMNPQRSTVWY" for aa in seq):
            logger.error(f"Skipping invalid sequence: {seq}")
            results.append((np.nan, np.nan, np.nan))
            continue

        pdb_str, plddt = _fold_seq_to_pdb_str(seq, esmfold_model)
        plddt = plddt[0]
        raw_if, dg = _score_pdb_str(pdb_str[0], chain_id)
        logger.info(f"Seq len={len(seq)} raw_IF={raw_if:.1f} ΔG={dg:.2f} kcal/mol pLDDT={plddt:.1f}")
        results.append((raw_if, dg, plddt))

    return results

def stability_score_batch(
    sequences: List[str],
    chain_id: str = "A",
    esmfold_model: str = "facebook/esmfold_v1"
) -> List[Tuple[float, float]]:
    """
    Batch absolute ΔG prediction:
      1) fold each seq with ESMFold
      2) score each structure with ESM-IF

    Returns list of (raw_IF_score, ΔG_kcal/mol).
    """
    _load_esmfold(esmfold_model)
    _load_if_model()

    results = []
    pdb_strings = []
    plddts = []

    _sequences = []

    start = time.time()
    for seq in sequences:
        if not seq or any(aa not in "ACDEFGHIKLMNPQRSTVWY" for aa in seq):
            logger.error(f"Skipping invalid sequence: {seq}")
            results.append((np.nan, np.nan))
            continue
        _sequences.append(seq)

    pdb_strs, plddts = _fold_seq_to_pdb_str(_sequences, esmfold_model)
    pdb_strings.extend(pdb_strs)
    plddts.extend(plddts)

    print(f"Time taken to fold sequences: {time.time() - start} seconds")

    start = time.time()
    results = _score_pdb_str_batch(pdb_strings, chain_id)
    print(f"Time taken to score sequences: {time.time() - start} seconds")

    # for pdb_str in pdb_strings:
    #     raw_if, dg = _score_pdb_str(pdb_str, chain_id)
    #     logger.info(f"Seq len={len(seq)} raw_IF={raw_if:.1f} ΔG={dg:.2f} kcal/mol")
    #     results.append((raw_if, dg))

    results = [(*e, plddt) for e, plddt in zip(results, plddts)]

    return results

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} SEQ1 [SEQ2 ...]")
        sys.exit(1)
    seqs = sys.argv[1:]
    scores = stability_score(seqs)
    for seq, (raw_if, dg, plddt) in zip(seqs, scores):
        print(f"{seq[:10]}… raw_IF={raw_if:.2f}, ΔG={dg:.2f} kcal/mol, pLDDT={plddt:.1f}")
