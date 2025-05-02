import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, AutoTokenizer, get_linear_schedule_with_warmup
from typing import Dict, Optional, Union, List
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from dataset import ZymCTRLDataset, ZymCTRLSFTDataset, ZymCTRLDPODataset
import random
import logging

import torch.nn.functional as F
import torch

from utils import perplexity_from_logits

def calculatePerplexity(input_ids, model, attention_mask):
    import math
    '''
    Computes perplexities for the generated sequences. 
    '''
    with torch.no_grad():
        # Ensure input_ids is 2D
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # Add batch dimension
        outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
    loss, logits = outputs[:2]
    return math.exp(loss)

class ZymCTRLModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        beta: float = 0.1,
        training_mode: str = "sft",
        max_length: int = 512,
        use_weighted_dpo: bool = False,
        weight_scale: float = 1.0
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)

        # add special tokens to tokenizer
        special_tokens_dict = {
            "additional_special_tokens": [
                "<stability=high>",
                "<stability=medium>",
                "<stability=low>"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)
            
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Make sure model knows about padding token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # Memory optimizations
        self.model.config.use_cache = False  # Disable KV cache for training
        self.model.gradient_checkpointing_enable()
        
        # Ensure model is in training mode
        self.model.train()
        
        # Training params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.beta = beta
        self.training_mode = training_mode
        self.max_length = max_length
        self.use_weighted_dpo = use_weighted_dpo
        self.weight_scale = weight_scale

    def forward(self, **inputs):
        return self.model(**inputs)

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute the appropriate loss based on the mode"""
        if self.training_mode == "sft":
            # Move tensors to device and pass to model
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['input_ids']
            )
            return outputs.loss
        else:
            # DPO loss
            return self._dpo_step(batch)
    
    def _compute_perplexity(self, batch: Dict[str, torch.Tensor]):
        import math
        '''
        Computes perplexity differences between chosen and rejected sequences.
        '''
        if self.training_mode == "dpo":
            chosen_logits = self.forward(input_ids=batch["chosen"]["input_ids"], attention_mask=batch["chosen"]["attention_mask"]).logits
            chosen_perplexity = perplexity_from_logits(chosen_logits, batch["chosen"]["input_ids"], batch["chosen"]["attention_mask"])

            # chosen_perplexity = calculatePerplexity(batch['chosen']['input_ids'], self.model, batch['chosen']['attention_mask'])
            
            print(f"PRE_DPO Chosen perplexity: {batch['chosen']['perplexity'].item()}")
            print(f"POST_DPO Chosen perplexity: {chosen_perplexity}")

            # Get perplexity for rejected sequence

            rejected_logits = self.forward(input_ids=batch["rejected"]["input_ids"], attention_mask=batch["rejected"]["attention_mask"]).logits
            rejected_perplexity = perplexity_from_logits(rejected_logits, batch["rejected"]["input_ids"], batch["rejected"]["attention_mask"])

            # rejected_perplexity = calculatePerplexity(batch['rejected']['input_ids'], self.model, batch['rejected']['attention_mask'])
            
            print(f"PRE_DPO Rejected perplexity: {batch['rejected']['perplexity'].item()}")
            print(f"POST_DPO Rejected perplexity: {rejected_perplexity}")
            
            # Calculate differences from batch perplexities
            chosen_diff = abs(chosen_perplexity - batch['chosen']['perplexity'].item())
            rejected_diff = abs(rejected_perplexity - batch['rejected']['perplexity'].item())

            print(f"Chosen diff: {chosen_diff}")
            print(f"Rejected diff: {rejected_diff}")

            return chosen_diff + rejected_diff
        else:
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['input_ids']
            )
            perplexity = math.exp(outputs.loss)

            diff = abs(perplexity - batch['perplexity'].item())

            return diff

    def training_step(self, batch, batch_idx):
        # Ensure we're in training mode
        self.model.train()
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        # Explicitly set eval mode for validation
        self.model.eval()
        loss = self._compute_loss(batch)
        perplexity = self._compute_perplexity(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("perplexity", perplexity, prog_bar=True, sync_dist=True)
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
        # Get policy and reference outputs
        chosen_outputs = self.model(
            input_ids=batch["chosen"]["input_ids"].to(self.device),
            attention_mask=batch["chosen"]["attention_mask"].to(self.device),
            labels=batch["chosen"]["input_ids"].to(self.device)
        )
        rejected_outputs = self.model(
            input_ids=batch["rejected"]["input_ids"].to(self.device),
            attention_mask=batch["rejected"]["attention_mask"].to(self.device),
            labels=batch["rejected"]["input_ids"].to(self.device)
        )
        
        # Compute log probabilities (negative loss is log probability)
        policy_chosen = -chosen_outputs.loss
        policy_rejected = -rejected_outputs.loss
        
        if self.use_weighted_dpo:
            # Get stability scores
            chosen_scores = batch["chosen"]["stability_score"].to(self.device)
            rejected_scores = batch["rejected"]["stability_score"].to(self.device)
            
            # Compute weights based on stability difference
            # Note: abs() because the difference direction depends on prefer_stable
            weights = torch.abs(chosen_scores - rejected_scores) * self.weight_scale
            
            # Compute weighted DPO loss
            loss = -weights * F.logsigmoid(self.beta * (policy_chosen - policy_rejected))
        else:
            # Standard DPO loss
            loss = -F.logsigmoid(self.beta * (policy_chosen - policy_rejected))
            
        return torch.mean(loss)

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
        batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        warmup_steps: int = 100,
        beta: float = 0.1,
        use_wandb: bool = False,
        use_weighted_dpo: bool = False,
        weight_scale: float = 1.0,
        weight_decay: float = 0.01,
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
        self.use_weighted_dpo = use_weighted_dpo
        self.weight_scale = weight_scale
        self.weight_decay = weight_decay
        self.trainer_kwargs = trainer_kwargs
        
        os.makedirs(output_dir, exist_ok=True)

    def train(
        self,
        train_dataset,
        val_dataset=None,
        test_dataset=None,
        num_epochs: int = 3,
        run_name: Optional[str] = None,
        training_mode: str = "sft"
    ):
        # Create model
        model = ZymCTRLModule(
            model_name=self.model_name,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            beta=self.beta,
            training_mode=training_mode,
            use_weighted_dpo=self.use_weighted_dpo,
            weight_scale=self.weight_scale
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
        if self.use_wandb:
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
                filename="{epoch}-{" + ("val_loss" if val_dataset else "train_loss") + ":.2f}",
                save_top_k=1,
                monitor="val_loss" if val_dataset else "train_loss",
                mode="min"
            ),
        ]

        # Add early stopping only if we have validation data
        if val_dataset:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=3,
                    mode="min"
                )
            )

        # Only add LearningRateMonitor if we have a logger
        if loggers:
            callbacks.append(LearningRateMonitor(logging_interval="step"))
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator="auto",
            devices=1,
            accumulate_grad_batches=self.gradient_accumulation_steps,
            logger=loggers,
            callbacks=callbacks,
            gradient_clip_val=1.0,
            precision="bf16-mixed",  # Enable mixed precision training with float16
            **self.trainer_kwargs
        )
        
        # Train
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
        
        # Save the final model in Hugging Face format
        output_dir = os.path.join(self.output_dir, "hf_model")
        os.makedirs(output_dir, exist_ok=True)
        model.model.save_pretrained(output_dir)
        model.tokenizer.save_pretrained(output_dir)
        
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
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--output_dir', type=str, default="checkpoints",
                        help='Directory to save checkpoints')
    parser.add_argument('--training_mode', type=str, default="sft",
                        help='Training mode: sft or dpo')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Beta parameter for DPO loss')
    parser.add_argument('--use_weighted_dpo', action='store_true',
                        help='Use weighted DPO with stability score differences as weights')
    parser.add_argument('--weight_scale', type=float, default=1.0,
                        help='Scaling factor for stability difference weights in weighted DPO')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--tag', type=str, default="",
                        help='Tag for the training data')
    parser.add_argument('--iteration_num', type=int, default=0,
                        help='Iteration number for the training data')
    parser.add_argument('--stability_threshold', type=float, default=1.0,
                        help='Minimum stability score difference for DPO pairs')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ZymCTRLTrainer(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
        beta=args.beta,
        use_wandb=not args.no_wandb,
        use_weighted_dpo=args.use_weighted_dpo,
        weight_scale=args.weight_scale,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay
    )
    
    # Initialize tokenizer for dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Add special tokens to tokenizer
    special_tokens_dict = {
        "additional_special_tokens": [
            "<stability=high>",
            "<stability=medium>",
            "<stability=low>"
        ]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    
    dataset_type = ZymCTRLSFTDataset if args.training_mode == "sft" else ZymCTRLDPODataset
    # Load datasets
    train_dataset = dataset_type(
        data_path=args.train_data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        training_mode=args.training_mode,
        stability_threshold=args.stability_threshold
    )
    
    val_dataset = None
    if args.val_data:
        val_dataset = dataset_type(
            data_path=args.val_data,
            tokenizer=tokenizer,
            max_length=args.max_length,
            training_mode=args.training_mode,
            stability_threshold=args.stability_threshold
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