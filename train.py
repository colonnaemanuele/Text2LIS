import os

from text2lis.model.args import args
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from process_data import get_synthetic_dataset
from text2lis.model.IterativeText2LIS import IterativeTextGuidedPoseGenerationModel
from inference import pred

from text2lis.model.tokenizer_ita import EnglishTokenizer
from text2lis.model.colator import zero_pad_collator

from text2lis.model.args import args


MIN_CONFIDENCE = 0.2
NUM_HAND_KEYPOINTS = 22
NUM_FACE_KEYPOINTS = 70


# ... (import statements and other initial code)
def get_optimizer(opt_str):
    if opt_str == "Adam":
        return Adam
    elif opt_str == "SGD":
        return SGD
    else:
        raise Exception("optimizer not supported. use Adam or SGD.")


def get_model_args(args, num_pose_joints, num_pose_dims):
    model_args = dict(
        tokenizer=EnglishTokenizer(),
        pose_dims=(num_pose_joints, num_pose_dims),
        hidden_dim=args["hidden_dim"],
        text_encoder_depth=args["text_encoder_depth"],
        pose_encoder_depth=args["pose_encoder_depth"],
        encoder_heads=args["encoder_heads"],
        max_seq_size=args["max_seq_size"],
        num_steps=args["num_steps"],
        tf_p=args["tf_p"],
        seq_len_weight=args["seq_len_weight"],
        noise_epsilon=args["noise_epsilon"],
        optimizer_fn=get_optimizer(args["optimizer"]),
        separate_positional_embedding=args["separate_positional_embedding"],
        encoder_dim_feedforward=args["encoder_dim_feedforward"],
        num_pose_projection_layers=args["num_pose_projection_layers"],
    )

    return model_args


class CustomErrorCallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Questo hook viene chiamato alla fine di ogni batch di addestramento
        pass

    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, RuntimeError) and "CUDA error: device-side assert triggered" in str(exception):
            print("Caught CUDA device-side assert error during training.")
            # Debug: Stampa lo stato del modello e dei dati
            print(f"Exception details: {exception}")

            # Puoi aggiungere altre azioni di debug qui, se necessario
            trainer.should_stop = False  # Ferma il training


if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    args = vars(args)
    if args["config_file"]:  # override args with yaml config file
        with open(args["config_file"], "r") as f:
            args = yaml.safe_load(f)

    args["batch_size"] = 1

    # Create synthetic datasets
    train_dataset, test_dataset = get_synthetic_dataset(num_samples=10000, max_seq_size=200, split_ratio=0.9)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, collate_fn=zero_pad_collator)
    # print(train_dataset[0]["pose"]["data"].shape)

    _, _, _, num_pose_joints, num_pose_dims = train_dataset[0]["pose"]["data"].shape

    model_args = get_model_args(args, num_pose_joints, num_pose_dims)

    model = IterativeTextGuidedPoseGenerationModel(**model_args)

    callbacks = []

    os.makedirs(f"./models/{args['model_name']}", exist_ok=True)
    callbacks.append(
        ModelCheckpoint(
            dirpath=f"./models/{args['model_name']}",
            filename="model",
            verbose=True,
            save_top_k=1,
            monitor="train_loss",
            mode="min",
        )
    )
    callbacks.append(CustomErrorCallback())

    trainer = pl.Trainer(
        max_epochs=args["max_epochs"],
        callbacks=callbacks,
        accelerator=(
            "gpu" if torch.cuda.is_available() else "cpu"
        ),  # Usa "cpu" se non hai una GPU
        log_every_n_steps=10,  # Frequenza di logging
        accumulate_grad_batches=1,  # Accumula i gradienti per questo numero di batch
    )

    trainer.fit(model, train_dataloaders=train_loader)

    # evaluate
    model = IterativeTextGuidedPoseGenerationModel.load_from_checkpoint(
        r"pretrained_models/checkpoints", **model_args
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # test seq_len_predictor
    diffs = []

    pred(
        model,
        test_dataset,
        os.path.join(f"./models/{args['model_name']}", args["output_dir"], "train"),
    )
    # pred(model, test_dataset, os.path.join(f"./models/{args['model_name']}", args['output_dir'], "test"))
