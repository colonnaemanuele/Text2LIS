import json
import os

import args as args
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from process_data import get_dataset

from tokenizer_ita import EnglishTokenizer

from args import args

from torch.optim import Adam, SGD
from text2lis.model.text2lis import TextGuidedPoseGenerationModel


def print_pose(pose_data, num_joints, num_dims):
    """
    Funzione di supporto per stampare i dati della posa.
    """
    # Verifica che ci sia almeno un frame
    if pose_data.shape[0] == 0:
        print("Nessun dato di posa disponibile.")
        return

    # Verifica che ci siano i joint
    if num_joints == 0:
        print("Nessun keypoint disponibile.")
        return

    for frame in range(pose_data.shape[0]):
        print(f"Frame {frame + 1}:")
        for joint in range(num_joints):
            joint_data = pose_data[frame, joint]  # Dati del joint per il frame corrente
            joint_info = ", ".join(
                [f"{d:.2f}" for d in joint_data]
            )  # Formatta i dati del joint
            print(f"  Joint {joint + 1}: {joint_info}")
        print("\n")


def draw_points_on_black_image(
    points,
    image_size=(640, 480),
    point_color=(0, 255, 0),
    point_radius=5,
    frame_number=1,
):
    black_image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    for x, y in points:
        cv2.circle(black_image, (int(x), int(y)), point_radius, point_color, -1)
    cv2.putText(
        black_image,
        f"Frame {frame_number}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return black_image


def convert_3d_to_2d(
    points_3d, focal_length=1.0, center_x=0, center_y=0, scale_factor=10
):
    points_2d = []
    for x, y, z in points_3d:
        x_2d = focal_length * x * scale_factor + center_x
        y_2d = focal_length * y * scale_factor + center_y
        points_2d.append((x_2d, y_2d))
    return points_2d


def visualize_3d_points(
    points_3d_list,
    image_size=(640, 480),
    focal_length=1.0,
    center_x=320,
    center_y=240,
    scale_factor=1000,
    video_filename="output_video.mp4",
):
    """
    Visualizza i punti 3D come un video di immagini nere, aggiunge le linee tra i punti e salva il video come file MP4.
    """
    # Crea un oggetto VideoWriter per salvare il video in formato MP4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec per MP4
    video_writer = cv2.VideoWriter(video_filename, fourcc, 30, image_size)  # 30 fps

    for i, points_3d in enumerate(points_3d_list):
        points_2d = convert_3d_to_2d(
            points_3d, focal_length, center_x, center_y, scale_factor
        )
        black_image = draw_points_on_black_image(
            points_2d, image_size, frame_number=i + 1
        )

        # Aggiungi il frame al video
        video_writer.write(black_image)

        # Mostra il frame corrente
        cv2.imshow("Points on Black Image", black_image)
        if cv2.waitKey(33) & 0xFF == ord("q"):
            break

    # Rilascia il VideoWriter e chiudi tutte le finestre
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Video salvato come {video_filename}")


def round_tensor(tensor, decimals=1):
    multiplier = 10**decimals
    return torch.round(tensor * multiplier) / multiplier


def save_tensor_to_file(tensor, filename):
    with open(filename, "w") as f:

        f.write(f"{tensor}\n")


def pred(
    model, dataset, output_dir, vis_process=False, gen_k=30, vis=True, subset=None
):
    os.makedirs(output_dir, exist_ok=True)
    print(dataset[0]["pose"]["data"].shape)
    # _, _, _, num_pose_joints, num_pose_dims = dataset[0]["pose"]["data"].shape
    pose_header = dataset[0]["pose"]["obj"].header
    preds = []

    model.eval()
    with torch.no_grad():
        for i, datum in enumerate(dataset):
            if subset is not None and datum["id"] not in subset:
                continue
            if i >= gen_k and subset is None:
                break

            first_pose = datum["initial"]
            seq_iter = model.forward(text=datum["text"], first_pose=first_pose)
            seq_list = list(seq_iter)
            stacked_tensor = torch.stack(seq_list, dim=0)

            if i == 0:
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                original_pose = datum["pose"]["data"]
                original_pose = original_pose.to(device).squeeze()
                stacked_tensor = stacked_tensor.to(device).squeeze()

                print(original_pose.size())
                print(stacked_tensor.size())
                if stacked_tensor.shape[0] > original_pose.shape[0]:
                    stacked_tensor = stacked_tensor[: original_pose.shape[0], :, :]
                else:
                    original_pose = original_pose[: stacked_tensor.shape[0], :, :]
                original_pose = torch.round(original_pose * 100) / 100
                stacked_tensor = torch.round(stacked_tensor * 100) / 100

                sq_error = torch.pow(original_pose - stacked_tensor, 2).sum(-1)

                print(f"Testo: {datum['text']}")
                print("Posa Originale:           Posa Generata:")
                print(f"{original_pose}    {stacked_tensor}")

                print("Errore Quadratico:")
                num_elements = sq_error.numel()
                mse = sq_error.sum() / num_elements
                print(f"Errore Quadratico Medio (MSE): {mse.item()}")
                difference_tensor = original_pose - stacked_tensor
                print("Tensore Differenza:")
                print(difference_tensor)

                save_tensor_to_file(stacked_tensor, "generato.txt")
                save_tensor_to_file(original_pose, "aggiustato.txt")

                visualize_3d_points(
                    original_pose,
                    image_size=(1920, 1080),
                    focal_length=1.0,
                    center_x=660,
                    center_y=580,
                    scale_factor=800,
                    video_filename="target.mp4",
                )

                print("Visualizzazione Posa Generata:")
                visualize_3d_points(
                    stacked_tensor,
                    image_size=(1920, 1080),
                    focal_length=2.0,
                    center_x=660,
                    center_y=580,
                    scale_factor=800,
                    video_filename="generata.mp4",
                )

    return preds


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


def get_optimizer(opt_str):
    if opt_str == "Adam":
        return Adam
    elif opt_str == "SGD":
        return SGD
    else:
        raise Exception("optimizer not supported. use Adam or SGD.")


if __name__ == "__main__":
    args = vars(args)
    model_args = get_model_args(args, 55, 3)
    model = TextGuidedPoseGenerationModel.load_from_checkpoint("demo\model.ckpt", **model_args)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # test seq_len_predictor
    diffs = []
    train_dataset, test_dataset = get_dataset(num_samples=10000, max_seq_size=200, split_ratio=0.9)

    pred(model, test_dataset, os.path.join(f"./models/{args['model_name']}", args["output_dir"], "train"))
