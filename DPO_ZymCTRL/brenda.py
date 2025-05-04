from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO
import io
import json

import asyncio
import aiohttp
from aiohttp import ClientSession
from tqdm.asyncio import tqdm as async_tqdm
import json

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

def get_brenda_dataloader(uniprot_ids, labels=None, batch_size=32, shuffle=True, transform=None):
  """
  Creates a DataLoader for Brenda enzyme sequences from UniProt.

  Args:
    uniprot_ids (list): List of UniProt IDs to fetch.
    labels (list, optional): List of labels corresponding to the UniProt IDs.
    batch_size (int): Number of samples per batch.
    shuffle (bool): Whether to shuffle the dataset.
    transform (callable, optional): Optional transform to be applied to the features.

  Returns:
    DataLoader: A PyTorch DataLoader for the dataset.
  """
  dataset = UniprotSequenceDataset(uniprot_ids, labels, transform=transform)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
  return dataloader, dataset

if __name__ == "__main__":
  # get_brenda_uniprot_ids(input_path="brenda.txt",
  # output_path="brenda_uniprot_ids.txt")
  # get_valid_brenda_uniprot_ids(intput_path="uniprot_accessions.txt")
  # asyncio.run(build_ec_protein_async("brenda.json"))
  # out = extract_ec_uniprot_pairs(brenda_file_path="brenda.json")
  # print(len(out))
  # dl, ds = get_brenda_dataloader(["P80225"])
  # Load and print first 10 key-value pairs from brenda.json
  with open("ec_protein_mapping.json", 'r') as file:
      brenda_data = json.load(file)
      print("First 10 key-value pairs from brenda.json:")
      for i, (key, value) in enumerate(brenda_data.items()):
          if i >= 10:
              break
          print(f"{key}: {value}")
  