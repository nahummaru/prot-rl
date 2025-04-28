import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, AutoTokenizer, get_linear_schedule_with_warmup
from typing import Dict, Optional
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os

class ZymCTRLModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        dpo_mode: bool = False
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Training params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.dpo_mode = dpo_mode

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        if self.dpo_mode:
            loss = self._dpo_step(batch)
        else:
            outputs = self.model(**batch)
            loss = outputs.loss

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def _dpo_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Get policy and reference outputs
        policy_outputs = self.model(**batch["chosen"])
        ref_outputs = self.model(**batch["rejected"])
        
        # Compute log probabilities
        policy_logprobs = -policy_outputs.loss
        ref_logprobs = -ref_outputs.loss
        
        # Compute DPO loss
        loss = -torch.mean(policy_logprobs - ref_logprobs)
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
        use_wandb: bool = True,
        **trainer_kwargs
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_steps = warmup_steps
        self.use_wandb = use_wandb
        self.trainer_kwargs = trainer_kwargs
        
        os.makedirs(output_dir, exist_ok=True)

    def train(
        self,
        train_dataset,
        val_dataset=None,
        num_epochs: int = 3,
        run_name: Optional[str] = None,
        dpo_mode: bool = False
    ):
        # Create model
        model = ZymCTRLModule(
            model_name=self.model_name,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            dpo_mode=dpo_mode
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
        if self.use_wandb:
            wandb_logger = WandbLogger(
                project="zymctrl-training",
                name=run_name,
                log_model=True
            )
            loggers.append(wandb_logger)
        
        # Set up checkpointing
        callbacks = [
            ModelCheckpoint(
                dirpath=self.output_dir,
                filename="{epoch}-{val_loss:.2f}",
                save_top_k=3,
                monitor="val_loss",
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
    # Example usage
    trainer = ZymCTRLTrainer(
        model_name="AI4PD/ZymCTRL",
        learning_rate=1e-5,
        batch_size=4,
        gradient_accumulation_steps=4,
        use_wandb=True
    )
    
    # For SFT:
    model = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=3,
        run_name="sft-run"
    )
    
    # For DPO:
    model = trainer.train(
        train_dataset=preference_dataset,
        val_dataset=val_dataset,
        num_epochs=3,
        run_name="dpo-run",
        dpo_mode=True
    )

if __name__ == "__main__":
    main() 