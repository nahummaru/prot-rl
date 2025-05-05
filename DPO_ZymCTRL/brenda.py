from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
import io
import json

import asyncio
import aiohttp
from aiohttp import ClientSession
from tqdm.asyncio import tqdm as async_tqdm
import json
import csv
import random
import torch

from stability import stability_score

def extract_ec_uniprot_pairs(brenda_file_path="brenda.txt"):
  out_data = {}
  # data = {}

  with open(brenda_file_path, 'r') as file:
    data = json.load(file)["data"]

  for id, value in data.items():
    # Check if ID is in EC format (e.g., "EC 1.1.1.1" or "1.1.1.1")
    if not (id.count(".") == 3 and all(part.isdigit() for part in id.split("."))):
      continue

    if "protein" not in value:
      continue
    
    for protein_id, protein_data in value["protein"].items():
      if "accessions" in protein_data:
        if id not in out_data:
          out_data[id] = []
        out_data[id].extend(protein_data["accessions"])

  # Write the results to a JSON file
  output_path = "ec_uniprot_mapping.json"
  with open(output_path, 'w') as outfile:
    json.dump(out_data, outfile, indent=2)

  # print(f"Saved EC to UniProt mapping with {len()ta} EC numbers to {output_path}")

  return out_data

async def get_protein_sequence_async(uniprot_id, session):
    """
    Asynchronously fetch a protein sequence for a specific UniProt ID.

    Args:
        uniprot_id (str): UniProt ID of the protein.
        session (ClientSession): An aiohttp ClientSession for making requests.

    Returns:
        str: The protein sequence if found, None otherwise.
    """
    try:
        url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
        async with session.get(url) as response:
            if response.status == 200:
                fasta_io = io.StringIO(await response.text())
                record = next(SeqIO.parse(fasta_io, "fasta"))
                return str(record.seq)
            else:
                print(f"Failed to fetch {uniprot_id}: HTTP status {response.status}")
                return None
    except Exception as e:
        print(f"Error processing {uniprot_id}: {e}")
        return None

async def build_ec_protein_async(brenda_path):
    """
    Asynchronously build a mapping of EC numbers to protein sequences.

    Args:
        brenda_path (str): Path to the BRENDA JSON file.

    Returns:
        dict: A dictionary mapping EC numbers to protein sequences.
    """
    uniprot = extract_ec_uniprot_pairs(brenda_file_path=brenda_path)

    data = {}
    async with aiohttp.ClientSession() as session:
        for key, value in async_tqdm(uniprot.items(), desc="Processing EC numbers"):
            tasks = [get_protein_sequence_async(id, session) for id in value]
            results = await asyncio.gather(*tasks)
            for id, sequence in zip(value, results):
                if sequence is not None:
                    if key not in data:
                        data[key] = []
                    data[key].append(sequence)

    output_path = "ec_protein_mapping.json"
    with open(output_path, 'w') as outfile:
        json.dump(data, outfile, indent=2)

    return data

def flatten_brenda_json_to_csv(input_path, output_path):
  """
  Flatten BRENDA JSON data to a CSV file with EC numbers and protein sequences.
  
  Args:
    input_path (str): Path to the input JSON file.
    output_path (str): Path to the output CSV file.
  """
  
  # Load the JSON data
  with open(input_path, 'r') as file:
    data = json.load(file)
  
  # Open CSV file for writing
  with open(output_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write header
    csv_writer.writerow(['EC_Number', 'Sequence'])
    
    # Write data rows
    for ec_number, sequences in data.items():
      for sequence in sequences:
        csv_writer.writerow([ec_number, sequence])
  
  print(f"Flattened data saved to {output_path}")

def generate_stability_labels(input_path, output_path, limit=None):
  """
  Load a CSV with EC numbers and protein sequences, add a stability score column,
  and save to a new CSV file.
  
  Args:
    input_path (str): Path to the input CSV file.
    output_path (str): Path to the output CSV file.
  """
  
  # Load the input CSV
  with open(input_path, 'r', newline='') as infile:
    reader = csv.reader(infile)
    header = next(reader)  # Get the header row
    rows = list(reader)    # Get all data rows
  
  # Add stability scores (random values between 0 and 1 for this example)
  batch_size = 512
  for i in range(0, len(rows), batch_size):
    if limit is not None and i == limit:
      break

    try:
      seqs = [rows[j][1] for j in range(i, i + batch_size)]
      stability_results = stability_score(seqs)
      for k, raw_if, stability in enumerate(stability_results):
        # raw_if, stability = stability_results[0]

        if stability < -2.0:  # More negative = more stable
          stability_label = "high"
        elif stability > 0.0:
          stability_label = "low"
        else:
          stability_label = "medium"

        rows[i + k].append(str(raw_if), str(stability), stability_label)

      torch.cuda.empty_cache()
    except Exception as e:
      print(f"Error generating stability annotation: {str(e)}")
      print(f"Skipping entry {i}")
  
  # Write to output CSV with the new column
  with open(output_path, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(header + ['raw_if', 'stability', 'stability_label'])
    writer.writerows(rows)
  
  print(f"Added stability labels to {len(rows)} sequences and saved to {output_path}")

if __name__ == "__main__":
  # get_brenda_uniprot_ids(input_path="brenda.txt",
  # output_path="brenda_uniprot_ids.txt")
  # get_valid_brenda_uniprot_ids(intput_path="uniprot_accessions.txt")
  # asyncio.run(build_ec_protein_async("brenda.json"))
  # out = extract_ec_uniprot_pairs(brenda_file_path="brenda.json")
  # print(len(out))
  # dl, ds = get_brenda_dataloader(["P80225"])
  # Load and print first 10 key-value pairs from brenda.json
  # with open("ec_protein_mapping.json", 'r') as file:
  #     brenda_data = json.load(file)
  #     print("First 10 key-value pairs from brenda.json:")
  #     for i, (key, value) in enumerate(brenda_data.items()):
  #         if i >= 10:
  #             break
  #         print(f"{key}: {value}")
  # flatten_brenda_json_to_csv("brenda/ec_protein_mapping.json",
  # "brenda/ec_sequences.csv")

  # add a limit for now to sanity check
  limit = 10
  generate_stability_labels("brenda/ec_sequences.csv", "brenda/ec_sequences_stability.csv", limit=limit)