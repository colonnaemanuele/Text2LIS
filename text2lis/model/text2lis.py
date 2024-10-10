import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

EPSILON = 1e-4
START_LEARNING_RATE = 1e-3
MAX_SEQ_LEN = 200

def masked_mse_loss(pose: torch.Tensor, pose_hat: torch.Tensor, confidence: torch.Tensor, model_num_steps: int = 10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pose = pose.to(device)
    pose_hat = pose_hat.to(device)
    sq_error = torch.pow(pose - pose_hat, 2).sum(-1)
    num_steps_norm = np.log(model_num_steps) ** 2 if model_num_steps != 1 else 1
    confidence = confidence.squeeze()
    return (sq_error * confidence).mean() * num_steps_norm

class TextGuidedPoseGenerationModel(nn.Module):
    def __init__(
            self,
            tokenizer,
            pose_dims: (int, int) = (137, 2),
            hidden_dim: int = 128,
            text_encoder_depth: int = 2,
            pose_encoder_depth: int = 4,
            encoder_heads: int = 2,
            encoder_dim_feedforward: int = 2048,
            max_seq_size: int = MAX_SEQ_LEN,
            min_seq_size: int = 20,
            num_steps: int = 10,
            tf_p: float = 0.5,
            noise_epsilon: float = EPSILON,
            seq_len_weight: float = 2e-8,
            separate_positional_embedding: bool = False,
            num_pose_projection_layers: int = 1,
            concat: bool = True,
            blend: bool = True
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.pose_dims = pose_dims
        self.hidden_dim = hidden_dim
        self.max_seq_size = max_seq_size
        self.min_seq_size = min_seq_size
        self.num_steps = num_steps
        self.tf_p = tf_p
        self.noise_epsilon = noise_epsilon
        self.seq_len_weight = seq_len_weight
        self.separate_positional_embedding = separate_positional_embedding
        self.concat = concat
        self.blend = blend

        pose_dim = int(np.prod(pose_dims))

        self.embedding = nn.Embedding(
            num_embeddings=len(tokenizer),
            embedding_dim=hidden_dim,
            padding_idx=tokenizer.pad_token_id,
        )

        self.step_embedding = nn.Embedding(
            num_embeddings=num_steps, embedding_dim=hidden_dim
        )

        if separate_positional_embedding:
            self.pos_positional_embeddings = nn.Embedding(
                num_embeddings=max_seq_size, embedding_dim=hidden_dim
            )
            self.text_positional_embeddings = nn.Embedding(
                num_embeddings=max_seq_size, embedding_dim=hidden_dim
            )
        else:
            self.positional_embeddings = nn.Embedding(
                num_embeddings=max_seq_size, embedding_dim=hidden_dim
            )
            self.alpha_pose = nn.Parameter(torch.randn(1))
            self.alpha_text = nn.Parameter(torch.randn(1))

        if num_pose_projection_layers == 1:
            self.pose_projection = nn.Linear(pose_dim, hidden_dim)
        else:
            self.pose_projection = nn.Sequential(
                nn.Linear(pose_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=encoder_heads,
                                                   dim_feedforward=encoder_dim_feedforward)

        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=text_encoder_depth)
        self.pose_encoder = nn.TransformerEncoder(encoder_layer, num_layers=pose_encoder_depth)

        self.step_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )

        self.seq_length = nn.Linear(hidden_dim, 1)

        self.pose_diff_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pose_dim),
        )

    def encode_text(self, texts):
        tokenized = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        if self.separate_positional_embedding:
            positional_embedding = self.text_positional_embeddings(torch.arange(tokenized['input_ids'].size(1)).to(self.device))
        else:
            positional_embedding = self.alpha_text * self.positional_embeddings(torch.arange(tokenized['input_ids'].size(1)).to(self.device))

        embedding = self.embedding(tokenized["input_ids"]) + positional_embedding
        encoded = self.text_encoder(embedding.transpose(0, 1),
                                    src_key_padding_mask=tokenized["attention_mask"].bool().logical_not()).transpose(0, 1)

        seq_length = self.seq_length(encoded).mean(axis=1)
        return {"data": encoded, "mask": tokenized["attention_mask"]}, seq_length+100

    def forward(self, text: str, first_pose: torch.Tensor, sequence_length: int = -1):
        text_encoding, seq_len = self.encode_text([text])
        seq_len = round(float(seq_len))
        seq_len = max(min(seq_len, self.max_seq_size), self.min_seq_size)
        sequence_length = seq_len if sequence_length == -1 else sequence_length

        first_pose = first_pose.unsqueeze(0)
        pose_sequence = {
            "data": torch.stack([first_pose] * sequence_length, dim=1),
            "mask": torch.zeros([1, sequence_length], dtype=torch.bool, device=self.device),
        }
        pose_sequence["data"] = pose_sequence["data"].view(1, sequence_length, 55, 3)

        if self.num_steps == 1:
            pred = self.refine_pose_sequence(pose_sequence, text_encoding)
            return pred
        else:
            for step_num in range(self.num_steps):
                pose_sequence["data"] = self.refinement_step(step_num, pose_sequence, text_encoding)[0]
            return pose_sequence["data"]

    def refinement_step(self, step_num, pose_sequence, text_encoding):
        batch_size = pose_sequence["data"].shape[0]
        pose_sequence["data"] = pose_sequence["data"].detach()

        batch_step_num = torch.full((batch_size, 1), step_num, dtype=torch.long, device=self.device)
        step_encoding = self.step_encoder(self.step_embedding(batch_step_num))

        change_pred = self.refine_pose_sequence(pose_sequence, text_encoding, step_encoding)

        cur_step_size = self.get_step_size(step_num + 1)
        prev_step_size = self.get_step_size(step_num) if step_num > 0 else 0
        step_size = cur_step_size - prev_step_size

        if self.blend:
            pred = (1 - step_size) * pose_sequence["data"] + step_size * change_pred
        else:
            pred = pose_sequence["data"] + step_size * change_pred

        return pred, cur_step_size

    def embed_pose(self, pose_sequence_data):
        batch_size, seq_length, _, _ = pose_sequence_data.shape
        flat_pose_data = pose_sequence_data.reshape(batch_size, seq_length, -1)

        positions = torch.arange(0, seq_length, dtype=torch.long, device=self.device)
        if self.separate_positional_embedding:
            positional_embedding = self.pos_positional_embeddings(positions)
        else:
            positional_embedding = self.alpha_pose * self.positional_embeddings(positions)

        pose_embedding = self.pose_projection(flat_pose_data) + positional_embedding
        return pose_embedding

    def encode_pose(self, pose_sequence, text_encoding, step_encoding=None):
        batch_size, seq_length, _, _ = pose_sequence["data"].shape

        pose_embedding = self.embed_pose(pose_sequence["data"])

        text_encoding["mask"] = torch.logical_not(text_encoding["mask"])

        if step_encoding is not None:
            step_mask = torch.zeros([step_encoding.size(0), 1], dtype=torch.bool, device=self.device)
            pose_text_sequence = torch.cat([pose_embedding, text_encoding["data"], step_encoding], dim=1)
            pose_text_mask = torch.cat([pose_sequence["mask"], text_encoding["mask"], step_mask], dim=1)
        else:
            pose_text_sequence = torch.cat([pose_embedding, text_encoding["data"]], dim=1)
            pose_text_mask = torch.cat([pose_sequence["mask"], text_encoding["mask"]], dim=1).bool()

        pose_text_sequence = (pose_text_sequence - pose_text_sequence.mean(dim=1, keepdim=True)) / pose_text_sequence.std(dim=1, keepdim=True)

        pose_encoding = self.pose_encoder(
            pose_text_sequence.transpose(0, 1), src_key_padding_mask=pose_text_mask
        ).transpose(0, 1)[:, :seq_length, :]

        return pose_encoding

    def refine_pose_sequence(self, pose_sequence, text_encoding, step_encoding=None):
        batch_size, seq_length, _, _ = pose_sequence["data"].shape
        pose_encoding = self.encode_pose(pose_sequence, text_encoding, step_encoding)

        flat_pose_projection = self.pose_diff_projection(pose_encoding)
        return flat_pose_projection.reshape(batch_size, seq_length, *self.pose_dims)

    def get_step_size(self, step_num):
        if step_num < 2:
            return 0.1
        else:
            return np.log(step_num) / np.log(self.num_steps)

    @property
    def device(self):
        return next(self.parameters()).device