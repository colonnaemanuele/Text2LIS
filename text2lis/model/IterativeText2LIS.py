from typing import List
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.optim import optimizer, Adam

EPSILON = 1e-4
START_LEARNING_RATE = 1e-3
MAX_SEQ_LEN = 200


def masked_mse_loss(
    pose: torch.Tensor,
    pose_hat: torch.Tensor,
    confidence: torch.Tensor,
    model_num_steps: int = 10,
):
    n = 3
    ##print(pose.size())
    ##print(pose_hat.size())
    ##print("Pose:")
    ##print(pose)
    ##print("\nPose_hat:")
    ##print(pose_hat)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pose = pose.to(device)
    pose_hat = pose_hat.to(device)
    # Loss by confidence. If missing joint, no loss. If less likely joint, less gradients.
    sq_error = torch.pow(pose - pose_hat, 2).sum(-1)
    num_steps_norm = (
        np.log(model_num_steps) ** 2 if model_num_steps != 1 else 1
    )  # normalization of the loss by the
    # model's step number
    ##print(sq_error.size())
    ##print(confidence.size())
    confidence = confidence.squeeze()
    ##print((sq_error * confidence).mean()* num_steps_norm)
    ##print(sq_error.size())
    ##print(confidence.size())

    return (sq_error * confidence).mean() * num_steps_norm


class IterativeTextGuidedPoseGenerationModel(pl.LightningModule):
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
        lr: float = START_LEARNING_RATE,
        noise_epsilon: float = EPSILON,
        seq_len_weight: float = 2e-8,
        optimizer_fn: optimizer = torch.optim.Adam,
        separate_positional_embedding: bool = False,
        num_pose_projection_layers: int = 1,
        concat: bool = True,
        blend: bool = True,
    ):
        super().__init__()
        self.lr = lr
        self.pose_encoder_depth = pose_encoder_depth
        self.noise_epsilon = noise_epsilon
        self.tf_p = tf_p
        self.encoder_heads = encoder_heads
        self.seq_len_weight = seq_len_weight
        self.encoder_dim_feedforward = encoder_dim_feedforward
        self.tokenizer = tokenizer
        self.max_seq_size = max_seq_size
        self.min_seq_size = min_seq_size
        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.pose_dims = pose_dims
        self.optimizer_fn = optimizer_fn
        self.separate_positional_embedding = separate_positional_embedding
        self.best_loss = np.inf
        self.concat = concat
        self.blend = blend
        self.num_pose_projection_layers = num_pose_projection_layers

        pose_dim = int(np.prod(pose_dims))

        # Embedding layers
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

            # positional embedding scalars
            self.alpha_pose = nn.Parameter(torch.randn(1))
            self.alpha_text = nn.Parameter(torch.randn(1))

        if num_pose_projection_layers == 1:
            self.pose_projection = nn.Linear(pose_dim, hidden_dim)
        else:  # Currently only supports 1 or 2 layers
            self.pose_projection = nn.Sequential(
                nn.Linear(pose_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        # encoding layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=encoder_heads,
            dim_feedforward=encoder_dim_feedforward,
        )

        self.text_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=text_encoder_depth
        )
        self.pose_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=pose_encoder_depth
        )

        # step encoder
        self.step_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Predict sequence length
        self.seq_length = nn.Linear(hidden_dim, 1)

        # Predict pose difference
        self.pose_diff_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, pose_dim),
        )

    def encode_text(self, texts: List[str]):
        ##print(texts[0])
        tokenized = self.tokenizer(texts, device=self.device)
        ##print(tokenized)

        if self.separate_positional_embedding:
            positional_embedding = self.text_positional_embeddings(
                tokenized["positions"]
            )
        else:
            positional_embedding = self.alpha_text * self.positional_embeddings(
                tokenized["positions"]
            )

        embedding = self.embedding(tokenized["tokens_ids"]) + positional_embedding
        encoded = self.text_encoder(
            embedding.transpose(0, 1), src_key_padding_mask=tokenized["attention_mask"]
        ).transpose(0, 1)

        seq_length = self.seq_length(encoded).mean(axis=1)
        return {"data": encoded, "mask": tokenized["attention_mask"]}, seq_length + 100

    def forward(self, text: str, first_pose: torch.Tensor, sequence_length: int = -1):
        # Encoding del testo
        text_encoding, seq_len = self.encode_text([text])
        seq_len = round(float(seq_len))
        seq_len = max(min(seq_len, self.max_seq_size), self.min_seq_size)
        sequence_length = seq_len if sequence_length == -1 else sequence_length

        # Preparazione della prima posa
        first_pose = first_pose.unsqueeze(0)
        # print(first_pose.shape)
        pose_sequence = {
            "data": torch.stack([first_pose] * sequence_length, dim=1),
            "mask": torch.zeros(
                [1, sequence_length], dtype=torch.bool, device=self.device
            ),
        }
        pose_sequence["data"] = pose_sequence["data"].view(1, sequence_length, 55, 3)

        if self.num_steps == 1:
            pred = self.refine_pose_sequence(pose_sequence, text_encoding)
            yield pred
        else:
            step_num = 0
            while True:
                # print(pose_sequence["data"])

                yield pose_sequence["data"][0].to(self.device)
                pose_sequence["data"] = self.refinement_step(
                    step_num, pose_sequence, text_encoding
                )[0]
                step_num += 1

    def refinement_step(self, step_num, pose_sequence, text_encoding):
        device = self.device

        batch_size = pose_sequence["data"].shape[0]
        pose_sequence["data"] = (
            pose_sequence["data"].detach().to(device)
        )  # Detach from graph and move to the same device as the model

        # Aggiungi il tipo LongTensor e sposta batch_step_num sulla stessa device del modello
        batch_step_num = (
            torch.repeat_interleave(torch.LongTensor([step_num]), batch_size)
            .unsqueeze(1)
            .to(device)
        )

        # Passa step_step_num alla step_embedding e alla step_encoder, che devono essere sullo stesso dispositivo
        step_encoding = self.step_encoder(self.step_embedding(batch_step_num))

        # Passa text_encoding alla refine_pose_sequence e cambia device di change_pred
        change_pred = self.refine_pose_sequence(
            pose_sequence, text_encoding, step_encoding
        ).to(device)

        cur_step_size = self.get_step_size(
            step_num + 1
        )  # Assicurati che cur_step_size sia sulla stessa device
        prev_step_size = (
            self.get_step_size(step_num) if step_num > 0 else 0
        )  # Assicurati che prev_step_size sia sulla stessa device
        step_size = cur_step_size - prev_step_size

        # Assicurati che tutti i tensor coinvolti siano sulla stessa device
        if self.blend:
            pred = (1 - step_size) * pose_sequence["data"] + step_size * change_pred
        else:
            pred = pose_sequence["data"] + step_size * change_pred  # add

        return pred, cur_step_size

    def embed_pose(self, pose_sequence_data):
        # print(pose_sequence_data.size())

        batch_size, seq_length, _, _ = pose_sequence_data.shape
        flat_pose_data = pose_sequence_data.reshape(batch_size, seq_length, -1).to(
            self.device
        )

        positions = torch.arange(0, seq_length, dtype=torch.long, device=self.device)
        if self.separate_positional_embedding:
            positional_embedding = self.pos_positional_embeddings(positions)
        else:
            alpha_pose = nn.Parameter(torch.randn(1)).to(self.device)
            positional_embeddings = nn.Embedding(
                num_embeddings=self.max_seq_size, embedding_dim=self.hidden_dim
            ).to(self.device)
            positional_embedding = alpha_pose * positional_embeddings(positions).to(
                self.device
            )

        pose_dim = int(np.prod(self.pose_dims))

        pose_projection = nn.Linear(pose_dim, self.hidden_dim).to(self.device)
        # print(flat_pose_data.size())
        # print(positional_embedding.size())

        pose_embedding = pose_projection(flat_pose_data) + positional_embedding
        ##########print(pose_embedding)
        return pose_embedding

    def encode_pose(self, pose_sequence, text_encoding, step_encoding=None):
        batch_size, seq_length, _, _ = pose_sequence["data"].shape

        # Encode pose sequence
        pose_embedding = self.embed_pose(pose_sequence["data"])
        text_encoding["mask"] = torch.logical_not(text_encoding["mask"])
        #########print(text_encoding["mask"])

        if step_encoding is not None:
            step_mask = torch.zeros(
                [step_encoding.size(0), 1], dtype=torch.bool, device=self.device
            )
            pose_text_sequence = torch.cat(
                [pose_embedding, text_encoding["data"], step_encoding], dim=1
            )
            pose_text_mask = torch.cat(
                [pose_sequence["mask"], text_encoding["mask"], step_mask], dim=1
            )
        else:
            pose_text_sequence = torch.cat(
                [pose_embedding, text_encoding["data"]], dim=1
            )

            # text_encoding["mask"][:, :4] = False  # Ad esempio, ignora gli ultimi token
            pose_text_mask = torch.cat(
                [pose_sequence["mask"], text_encoding["mask"]], dim=1
            ).bool()
            #########print(pose_text_mask)

        # Normalizzazione degli input
        pose_text_sequence = (
            pose_text_sequence - pose_text_sequence.mean(dim=1, keepdim=True)
        ) / pose_text_sequence.std(dim=1, keepdim=True)

        # Verifica dei dati concatenati

        # Encoder Layer and Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.encoder_heads,
            dim_feedforward=self.encoder_dim_feedforward,
        ).to(self.device)
        pose_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.pose_encoder_depth
        ).to(self.device)

        pose_encoding = pose_encoder(
            pose_text_sequence.transpose(0, 1), src_key_padding_mask=pose_text_mask
        ).transpose(0, 1)[:, :seq_length, :]
        layer_norm = nn.LayerNorm(self.hidden_dim).to(self.device)

        # Normalizzazione dell'output
        # pose_encoding = layer_norm(pose_encoding)

        #########print(pose_encoding)
        return pose_encoding

    def __get_text_pose_encoder(self):
        if hasattr(self, "text_pose_encoder"):
            return self.text_pose_encoder
        else:
            return self.pose_encoder

    def refine_pose_sequence(self, pose_sequence, text_encoding, step_encoding=None):
        batch_size, seq_length, _, _ = pose_sequence["data"].shape
        pose_encoding = self.encode_pose(pose_sequence, text_encoding, step_encoding)

        # Predict desired change
        flat_pose_projection = self.pose_diff_projection(pose_encoding)
        return flat_pose_projection.reshape(batch_size, seq_length, *self.pose_dims)

    def get_step_size(self, step_num):
        if step_num < 2:
            return 0.1
        else:
            return np.log(step_num) / np.log(self.num_steps)

    def training_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, phase="train")

    def validation_step(self, batch, *unused_args):
        return self.step(batch, *unused_args, phase="validation")

    def step(self, batch, *unused_args, phase: str):
        """
        @param batch: data batch
        @param phase: either "train" or "validation"

        """

        text_encoding, sequence_length = self.encode_text(batch["text"])
        pose = batch["pose"]

        ##print(pose["length"])
        initial = batch["initial"]
        # print(pose["data"].shape)

        batch_size, _, _, _, _, _ = pose["data"].shape
        ##print(initial.shape)
        pose["data"] = pose["data"].squeeze()
        # pose["data"] = pose["data"].unsqueeze(0)
        ##print(pose["data"][:, 0].size())
        ##print("posa originale")

        ##print(initial.shape)
        # print(batch["initial"].shape)

        # Repeat the first frame for initial prediction
        batch_size, pose_seq_length, num_keypoints, _ = pose["data"].shape

        pose_sequence = {
            "data": torch.stack([batch["initial"]] * pose_seq_length, dim=1),
            "mask": torch.logical_not(pose["inverse_mask"]),
        }
        ##print("primo frame")

        pose_sequence["data"] = pose_sequence["data"].view(
            batch_size, pose_seq_length, 55, 3
        )

        ##print(pose_sequence["mask"].size())

        if self.num_steps == 1:
            pred = self.refine_pose_sequence(pose_sequence, text_encoding)
            l1_gold = pose["data"]
            refinement_loss = masked_mse_loss(
                l1_gold, pred, pose["confidence"], self.num_steps
            )
            ##print(refinement_loss)
        else:
            ##print(batch["text"])

            refinement_loss = 0
            for i in range(self.num_steps):
                pred, step_size = self.refinement_step(i, pose_sequence, text_encoding)
                l1_gold = (
                    step_size * pose["data"] + (1 - step_size) * pose_sequence["data"]
                )
                ##print(l1_gold.size())
                ##print(pred.size())
                refinement_loss += masked_mse_loss(
                    l1_gold, pred, pose["confidence"], self.num_steps
                )
                ##print(refinement_loss)

                teacher_forcing_step_level = np.random.rand(1)[0] < self.tf_p
                pose_sequence["data"] = (
                    l1_gold
                    if phase == "validation" or teacher_forcing_step_level
                    else pred
                )

                if phase == "train":  # add just a little noise while training
                    pose_sequence["data"] = (
                        pose_sequence["data"]
                        + torch.randn_like(pose_sequence["data"]) * self.noise_epsilon
                    )

        sequence_length_loss = F.mse_loss(sequence_length, pose["length"])
        loss = refinement_loss + self.seq_len_weight * sequence_length_loss
        # print(refinement_loss)
        # print(sequence_length_loss)
        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log(
            phase + "_seq_length_loss", sequence_length_loss, batch_size=batch_size
        )
        self.log(phase + "_refinement_loss", refinement_loss, batch_size=batch_size)
        self.log(phase + "_loss", loss, batch_size=batch_size)

        return refinement_loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)
