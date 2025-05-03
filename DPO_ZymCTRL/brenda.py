import torch
from torch.utils.data import Dataset, DataLoader
import requests
from Bio import SeqIO
import io
import re
from tqdm import tqdm

class UniprotSequenceDataset(Dataset):
  def __init__(self, uniprot_ids, labels=None, transform=None):
    """
    Args:
      uniprot_ids (list): List of UniProt IDs to fetch.
      labels (list, optional): List of labels corresponding to the UniProt IDs.
      transform (callable, optional): Optional transform to be applied on a sample.
    """
    self.uniprot_ids = uniprot_ids
    self.labels = labels
    self.transform = transform
    self.sequences = {}
    self.ec_numbers = {}
    self._fetch_sequences_and_ec()
    
  def _fetch_sequences_and_ec(self):
    retained_uniprot_ids = []
    for uniprot_id in tqdm(self.uniprot_ids):
      try:
        url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
        response = requests.get(url)
        if response.status_code == 200:
          fasta_io = io.StringIO(response.text)
          record = next(SeqIO.parse(fasta_io, "fasta"))

          print("record", record)
          
          # Extract EC number from the record description
          description = record.description
          ec_match = [x for x in description.split() if x.startswith('EC=')]
          if ec_match:
            self.sequences[uniprot_id] = str(record.seq)
            self.ec_numbers[uniprot_id] = ec_match[0].split('=')[1]
            retained_uniprot_ids.append(uniprot_id)
          else:
            print(f"Skipping {uniprot_id}: No EC number found")
        else:
          print(f"Failed to fetch {uniprot_id}: {response.status_code}")
      except Exception as e:
        print(f"Error processing {uniprot_id}: {e}")
    
    # Update uniprot_ids list to only contain entries with EC numbers
    self.uniprot_ids = retained_uniprot_ids
    
    # Update labels if they exist
    if self.labels is not None:
      retained_indices = [i for i, uid in enumerate(self.uniprot_ids) 
                         if uid in retained_uniprot_ids]
      self.labels = [self.labels[i] for i in retained_indices]

  def filter_multi_ec_entries(self):
    """
    Removes UniProt entries that have multiple EC numbers (indicated by semicolons).
    Returns the number of entries removed.
    """
    entries_to_remove = []
    for uniprot_id, ec in self.ec_numbers.items():
      if ';' in ec:
        entries_to_remove.append(uniprot_id)
    
    # Remove the identified entries
    for uniprot_id in entries_to_remove:
      self.uniprot_ids.remove(uniprot_id)
      del self.sequences[uniprot_id]
      del self.ec_numbers[uniprot_id]
    
    # Update labels if they exist
    if self.labels is not None:
      # Create mapping of uniprot_id to index
      id_to_idx = {id: idx for idx, id in enumerate(self.uniprot_ids)}
      self.labels = [self.labels[i] for i in range(len(self.labels)) 
              if self.uniprot_ids[i] not in entries_to_remove]
    
    return len(entries_to_remove)

  def __len__(self):
    return len(self.uniprot_ids)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    uniprot_id = self.uniprot_ids[idx]
    sequence = self.sequences[uniprot_id]
    ec_number = self.ec_numbers[uniprot_id]
    
    # Convert sequence to numerical features
    features = [ord(aa) for aa in sequence]
    features = torch.tensor(features, dtype=torch.float32)
    
    if self.transform:
      features = self.transform(features)
      
    if self.labels is not None:
      label = torch.tensor(self.labels[idx], dtype=torch.long)
      return features, label, ec_number
    else:
      return features, ec_number

def get_brenda_uniprot_ids(input_path="brenda_download.txt", output_path=None):
  # Regular expression pattern to match UniProt accession codes
  uniprot_pattern = re.compile(r'\b[A-NR-Z][0-9][A-Z0-9]{3}[0-9]\b')

  # Set to store unique UniProt accession codes
  uniprot_accessions = set()

  # Read and parse the BRENDA flat file
  with open(input_path, 'r') as file:
    for line in file:
      matches = uniprot_pattern.findall(line)
      uniprot_accessions.update(matches)

  # Optionally, write the accession codes to a file
  if output_path is not None:
    with open(output_path, 'w') as output_file:
      for accession in sorted(uniprot_accessions):
        output_file.write(f"{accession}\n")

  return uniprot_accessions

def get_valid_brenda_uniprot_ids(intput_path="accession_ids.txt"):
  # Read accession IDs from the file
  with open(intput_path, 'r') as file:
    uniprot_ids = [line.strip() for line in file if line.strip()]

  uniprot_ids = uniprot_ids[:100]

  # Create a dataset to fetch the sequences and EC numbers
  dataset = UniprotSequenceDataset(uniprot_ids)
  
  # Filter out entries with multiple EC numbers
  removed_count = dataset.filter_multi_ec_entries()
  
  # Get the remaining valid IDs
  valid_ids = set(dataset.uniprot_ids)
  
  # Save the valid IDs to a new file
  output_path = intput_path.replace('.txt', '_valid.txt')
  with open(output_path, 'w') as output_file:
    for accession in sorted(valid_ids):
      ec_number = dataset.ec_numbers[accession]
      output_file.write(f"{accession}\t{ec_number}\n")
  
  print(f"Processed {len(uniprot_ids)} IDs. Removed {removed_count} with multiple EC numbers.")
  print(f"Saved {len(valid_ids)} valid IDs to {output_path}")
  
  return valid_ids

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
  get_valid_brenda_uniprot_ids(intput_path="uniprot_accessions.txt")
  # dl, ds = get_brenda_dataloader(["P80225"])
  # breakpoint()