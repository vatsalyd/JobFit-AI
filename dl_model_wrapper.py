# dl_model_wrapper.py
import torch
from transformers import AutoTokenizer, AutoModel
from torch import nn


class ResumeMatchModel(nn.Module):
    """Dual-encoder SBERT model with a deeper regression head."""

    def __init__(self, model_name: str):
        super(ResumeMatchModel, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        # Deeper head: 768*2 → 512 → 128 → 1  (was 768*2 → 256 → 1)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, resume_input_ids, resume_attention_mask, jd_input_ids, jd_attention_mask):
        r_out = self.encoder(input_ids=resume_input_ids, attention_mask=resume_attention_mask).pooler_output
        j_out = self.encoder(input_ids=jd_input_ids, attention_mask=jd_attention_mask).pooler_output
        combined = torch.cat((r_out, j_out), dim=1)
        return self.regressor(combined).squeeze(-1)


class DLModelWrapper:
    """Inference wrapper for the fine-tuned deep-learning model."""

    MAX_LEN = 384  # Increased from 256 to capture more resume content

    def __init__(self, model_dir: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ResumeMatchModel(model_name).to(self.device)
        self.model.load_state_dict(
            torch.load(f"{model_dir}/dl_resume_match.pt", map_location=self.device, weights_only=True)
        )
        self.model.eval()

    def predict(self, resume_text: str, jd_text: str) -> float:
        enc_resume = self.tokenizer(
            resume_text,
            truncation=True,
            padding="max_length",
            max_length=self.MAX_LEN,
            return_tensors="pt",
        )
        enc_jd = self.tokenizer(
            jd_text,
            truncation=True,
            padding="max_length",
            max_length=self.MAX_LEN,
            return_tensors="pt",
        )

        with torch.no_grad():
            score = self.model(
                enc_resume["input_ids"].to(self.device),
                enc_resume["attention_mask"].to(self.device),
                enc_jd["input_ids"].to(self.device),
                enc_jd["attention_mask"].to(self.device),
            ).item()
        return round(score * 100, 1)
