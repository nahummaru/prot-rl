import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, AutoTokenizer, get_linear_schedule_with_warmup
from typing import Dict, Optional, Union, List
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.nn.functional as F
import pandas as pd
from pathlib import Path


# NOTE: This v0 versino of this. 
class ZymCTRLDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 512,
        training_mode: str = "sft"
    ):
        self.tokenizer = tokenizer

        import pdb; pdb.set_trace()
        # Set up padding token if not set. 
        # NOTE: Make sure that this is not problematic. Could be issue for DPO paired data where samples are diff lengths
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.max_length = max_length
        self.training_mode = training_mode
        
        # Load data
        self.data = pd.read_csv(data_path)
        
        if training_mode == "dpo":
            # TODO: Process data to create paired sequences for DPO. 
            # For DPO, we expect columns: sequence, stability_score

            # Notes: 
            # For DPO, we will create paried sequences.
            # It will contrast a high stability sequence with a low stability sequence.

            assert "sequence" in self.data.columns
            assert "stability_score" in self.data.columns
            
            # Convert stability scores to weights for DPO
            self.weights = torch.tensor(self.data["stability_score"].values, dtype=torch.float32)
        else:
            # For SFT, we just need the sequence column

            # NOTES:
            # For SFT, we will select the most stable and least stable sequences for training.
            # We then create an input with a corresponding control tag <stability=high> or <stability=low>
            # We can then train the model so that it learns to conditionally generate enzymes with the correct stability.

            assert "sequence" in self.data.columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data["sequence"].iloc[idx]
        
        # Tokenize sequence
        inputs = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Squeeze out the batch dimension since DataLoader will handle batching
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        if self.training_mode == "dpo":
            # For DPO, include the weight
            inputs["weight"] = self.weights[idx]
            
        return inputs

class ZymCTRLModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        beta: float = 0.1,
        training_mode: str = "sft",
        max_length: int = 512
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        # Set up padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        # Make sure model knows about padding token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Training params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.beta = beta
        self.training_mode = training_mode
        self.max_length = max_length

    def forward(self, **inputs):
        return self.model(**inputs)

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the appropriate loss based on the mode"""
        if self.training_mode == "sft":
            # Standard language modeling loss
            outputs = self.model(**batch)
            return outputs.loss
        else:
            # DPO loss
            return self._dpo_step(batch)

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def _compute_logprobs(self, sequences: Union[str, List[str]]) -> torch.Tensor:
        """Compute log probabilities for sequences"""
        if isinstance(sequences, str):
            sequences = [sequences]
            
        all_log_probs = []
        for sequence in sequences:
            inputs = self.tokenizer(
                sequence,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            neg_log_likelihood = outputs.loss
            all_log_probs.append(-neg_log_likelihood.unsqueeze(0))
            
        return torch.cat(all_log_probs)

    def _dpo_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "chosen" in batch and "rejected" in batch:
            # Paired DPO
            policy_chosen = self._compute_logprobs(batch["chosen"])
            policy_rejected = self._compute_logprobs(batch["rejected"])
            
            loss = -F.logsigmoid(self.beta * (policy_chosen - policy_rejected))
            return torch.mean(loss)
        else:
            # Weighted DPO
            sequences = batch["sequence"]
            weights = batch["weight"].to(self.device)
            policy_logprobs = self._compute_logprobs(sequences)
            
            # Normalize weights
            weights = torch.softmax(weights, dim=0)
            
            # Compute weighted DPO loss
            loss = F.cross_entropy(self.beta * policy_logprobs, weights)
            return loss

    def configure_optimizers(self):
        # Set up optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Set up scheduler
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            }
        }

class ZymCTRLTrainer:
    def __init__(
        self,
        model_name: str,
        output_dir: str = "checkpoints",
        learning_rate: float = 1e-5,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 100,
        beta: float = 0.1,
        use_wandb: bool = False,
        **trainer_kwargs
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.beta = beta
        self.use_wandb = use_wandb
        self.trainer_kwargs = trainer_kwargs
        
        os.makedirs(output_dir, exist_ok=True)

    def train(
        self,
        train_dataset,
        val_dataset=None,
        num_epochs: int = 3,
        run_name: Optional[str] = None,
        training_mode: str = "sft"
    ):
        # Create model
        model = ZymCTRLModule(
            model_name=self.model_name,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            beta=self.beta,
            training_mode=training_mode
        )
        
        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_dataloader = None
        if val_dataset:
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                num_workers=4
            )
        
        # Set up logging
        loggers = []
        # TODO: Remove this. Figure out how to turn off wandb logging by cmd line
        if self.use_wandb and False:
            wandb_logger = WandbLogger(
                project="zymctrl-training",
                name=run_name,
                log_model=True
            )
            loggers.append(wandb_logger)
        
        # Set up callbacks
        callbacks = [
            ModelCheckpoint(
                dirpath=self.output_dir,
                filename="{epoch}-{val_loss:.2f}",
                save_top_k=3,
                monitor="val_loss",
                mode="min"
            ),
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(
                monitor="val_loss",
                patience=3,
                mode="min"
            )
        ]
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator="auto",
            devices=1,
            accumulate_grad_batches=self.gradient_accumulation_steps,
            logger=loggers,
            callbacks=callbacks,
            gradient_clip_val=1.0,
            **self.trainer_kwargs
        )
        
        # Train
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
        
        return model

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ZymCTRL using PyTorch Lightning')
    parser.add_argument('--model_name', type=str, default="AI4PD/ZymCTRL",
                        help='Name or path of the model to train')
    parser.add_argument('--train_data', type=str, required=True,
                        help='Path to training data CSV')
    parser.add_argument('--val_data', type=str,
                        help='Path to validation data CSV')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--output_dir', type=str, default="checkpoints",
                        help='Directory to save checkpoints')
    parser.add_argument('--training_mode', type=str, default="sft",
                        help='Training mode: sft or dpo')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Beta parameter for DPO loss')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ZymCTRLTrainer(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
        beta=args.beta,
        use_wandb=not args.no_wandb
    )
    
    # Initialize tokenizer for dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load datasets
    train_dataset = ZymCTRLDataset(
        data_path=args.train_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        training_mode=args.training_mode
    )
    
    val_dataset = None
    if args.val_data:
        val_dataset = ZymCTRLDataset(
            data_path=args.val_data,
            tokenizer=tokenizer,
            max_length=args.max_length,
            training_mode=args.training_mode
        )
    
    # Train model
    model = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=args.num_epochs,
        run_name=f"{'dpo' if args.training_mode == 'dpo' else 'sft'}-run",
        training_mode=args.training_mode
    )

if __name__ == "__main__":
    main() 