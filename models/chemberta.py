"""Pretrained ChemBERTa encoder with [CLS], Mean, or Attention pooling and 2-stage unfreezing."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem

from models.readouts import build_readout


def clean_smiles(smiles: str) -> str:
    """Strips salts, counterions, and extra fragments, keeping the largest molecular fragment."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        frags = Chem.GetMolFrags(mol, asMols=True)
        return Chem.MolToSmiles(max(frags, key=lambda m: m.GetNumAtoms()))
    except Exception:
        return smiles


class AttentionPool(nn.Module):
    """Multi-head attention pooling over non-padding SMILES tokens."""
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, num_heads)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D], mask: [B, L]
        scores = self.attn(x).masked_fill(mask.unsqueeze(-1) == 0, -1e9)  
        weights = torch.softmax(scores, dim=1)                             
        return torch.einsum('blh,bld->bhd', weights, x).mean(dim=1)       


class SMILEDataset(Dataset):
    def __init__(self, smiles_list, targets, tokenizer, max_length=128, bounds=None):
        self.smiles_list = [clean_smiles(s) for s in smiles_list]
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.bounds = bounds

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, i):
        encoding = self.tokenizer(
            self.smiles_list[i],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'y': torch.tensor(self.targets[i], dtype=torch.float32),
        }
        if self.bounds is not None:
            item['bounds'] = torch.tensor(self.bounds[i], dtype=torch.float32)
        return item


class ChemBERTa(nn.Module):
    def __init__(self, cfg, readout, n_targets, data_dir, device):
        super().__init__()
        self.cfg = cfg
        self.data_dir = data_dir
        self.hf_model_name = cfg.checkpoint
        self.max_length = cfg.max_length
        self.pool_type = cfg.get('pool', 'mean')          # 'mean' | 'cls' | 'attention'
        self.two_stage = bool(cfg.get('two_stage', False))
        self.unfreeze_epoch = int(cfg.get('unfreeze_epoch', 5))
        self._encoder_frozen = bool(cfg.freeze_encoder or self.two_stage)

        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
        self.encoder = AutoModel.from_pretrained(self.hf_model_name).to(device)

        if not self._encoder_frozen and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()
        elif self._encoder_frozen and hasattr(self.encoder, "gradient_checkpointing_disable"):
            self.encoder.gradient_checkpointing_disable()

        self.encoder.requires_grad_(not self._encoder_frozen)
        self.embedding_dim = self.encoder.config.hidden_size

        if self.pool_type == 'attention':
            self.attn_pool = AttentionPool(
                self.embedding_dim, num_heads=int(cfg.get('pool_heads', 4))
            ).to(device)

        self.readout = build_readout(readout, self.embedding_dim, n_targets).to(self.encoder.device)

    @property
    def device(self):
        return self.encoder.device

    def set_epoch(self, epoch: int, optimizer: torch.optim.Optimizer):
        """Hook called by finetune.py each epoch; manages 2-stage unfreezing via lr_scale."""
        if not self.two_stage:
            return
        should_freeze = epoch <= self.unfreeze_epoch
        if should_freeze != self._encoder_frozen:
            self._encoder_frozen = should_freeze
            self.encoder.requires_grad_(not should_freeze)
            if should_freeze:
                self.encoder.eval()
                if hasattr(self.encoder, "gradient_checkpointing_disable"):
                    self.encoder.gradient_checkpointing_disable()
            else:
                self.encoder.train()
                if hasattr(self.encoder, "gradient_checkpointing_enable"):
                    self.encoder.gradient_checkpointing_enable()
            for group in optimizer.param_groups:
                if group.get('name') == 'encoder':
                    group['lr_scale'] = 0.0 if should_freeze else 1.0
            state = "FROZEN" if should_freeze else "UNFROZEN"
            print(f"[Epoch {epoch}] 2-stage schedule -> Encoder {state}", flush=True)

    def train(self, mode=True):
        super().train(mode)
        if self._encoder_frozen:
            self.encoder.eval()
        return self

    def fit_preprocessing(self, train):
        pass

    def loader(self, frame, targets, batch_size, shuffle=False):
        smiles_list = frame['SMILES'].tolist()
        y_values = frame[targets].values


        bounds = None
        if targets and (targets[0] + '_conf_low') in frame.columns:
            low = frame[[t + '_conf_low' for t in targets]].to_numpy(dtype=np.float32)
            high = frame[[t + '_conf_high' for t in targets]].to_numpy(dtype=np.float32)
            bounds = np.stack([low, high], axis=-1)

        dataset = SMILEDataset(
            smiles_list=smiles_list,
            targets=y_values,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            bounds=bounds,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def param_groups(self):
        readout_params = list(self.readout.parameters())
        if hasattr(self, 'attn_pool'):
            readout_params += list(self.attn_pool.parameters())

        groups = [{
            'name': 'readout',
            'params': readout_params,
            'lr': self.cfg.readout_lr,
            'lr_scale': 1.0,
        }]

        if not self.cfg.freeze_encoder or self.two_stage:
            groups.append({
                'name': 'encoder',
                'params': list(self.encoder.parameters()),
                'lr': self.cfg.encoder_lr,
                'lr_scale': 0.0 if self._encoder_frozen else 1.0,
            })

        return groups

    def forward(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state

        if self.pool_type == 'cls':
            pooled_embedding = token_embeddings[:, 0, :]
        elif self.pool_type == 'attention':
            pooled_embedding = self.attn_pool(token_embeddings, attention_mask)
        else:
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled_embedding = sum_embeddings / sum_mask

        return self.readout(pooled_embedding)