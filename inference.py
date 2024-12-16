import os
import cv2
import numpy as np
import torch
from torch.optim import Adam, SGD
from text2lis.model.text2lis import TextGuidedPoseGenerationModel
from text2lis.model.process_data import get_dataset
from text2lis.model.tokenizer_ita import EnglishTokenizer


def print_pose(pose_data, num_joints):
    """
    Prints the pose data for each frame and joint.

    Args:
        pose_data (numpy.ndarray): A 3D array of shape (num_frames, num_joints, num_dimensions)
                                   containing the pose data for each frame and joint.
        num_joints (int): The number of joints in the pose data.
    """
    if pose_data.shape[0] == 0:
        print("No pose data available.")
        return
    if num_joints == 0:
        print("No keypoints available.")
        return
    for frame in range(pose_data.shape[0]):
        print(f"Frame {frame + 1}:")
        for joint in range(num_joints):
            joint_data = pose_data[frame, joint]
            joint_info = ", ".join([f"{d:.2f}" for d in joint_data])
            print(f"  Joint {joint + 1}: {joint_info}")
        print("\n")


def draw_points_on_black_image(
    points,
    image_size=(640, 480),
    point_color=(0, 255, 0),
    point_radius=5,
    frame_number=1,
):
    """
    Draws points on a black image and annotates it with the frame number.

    Args:
        points (list of tuples): List of (x, y) coordinates for the points to be drawn.
        image_size (tuple, optional): Size of the image as (width, height). Defaults to (640, 480).
        point_color (tuple, optional): Color of the points in BGR format. Defaults to (0, 255, 0).
        point_radius (int, optional): Radius of the points. Defaults to 5.
        frame_number (int, optional): Frame number to be annotated on the image. Defaults to 1.

    Returns:
        numpy.ndarray: Image with points and frame number annotated.
    """
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
    points_3d,
    focal_length=1.0,
    center_x=0,
    center_y=0,
    scale_factor=10,
):
    """
    Convert a list of 3D points to 2D points using a simple projection.

    Args:
        points_3d (list of tuples): A list of tuples where each tuple represents a point in 3D space (x, y, z).
        focal_length (float, optional): The focal length of the projection. Defaults to 1.0.
        center_x (int, optional): The x-coordinate of the center of the projection. Defaults to 0.
        center_y (int, optional): The y-coordinate of the center of the projection. Defaults to 0.
        scale_factor (int, optional): A scaling factor to apply to the projected points. Defaults to 10.

    Returns:
        list of tuples: A list of tuples where each tuple represents a point in 2D space (x_2d, y_2d).
    """
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
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_filename, fourcc, 30, image_size)

    for i, points_3d in enumerate(points_3d_list):
        points_2d = convert_3d_to_2d(
            points_3d, focal_length, center_x, center_y, scale_factor
        )
        black_image = draw_points_on_black_image(
            points_2d, image_size, frame_number=i + 1
        )
        video_writer.write(black_image)
        cv2.imshow("Points on Black Image", black_image)
        if cv2.waitKey(33) & 0xFF == ord("q"):
            break

    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Video saved as {video_filename}")


def round_tensor(tensor, decimals=1):
    multiplier = 10**decimals
    return torch.round(tensor * multiplier) / multiplier


def save_tensor_to_file(tensor, filename):
    with open(filename, "w") as f:
        f.write(f"{tensor}\n")


def pred(
    model,
    dataset,
    output_dir,
    vis_process=False,
    gen_k=30,
    vis=True,
    subset=None,
):
    os.makedirs(output_dir, exist_ok=True)
    preds = []

    model.eval()
    with torch.no_grad():
        for i, datum in enumerate(dataset):
            if subset is not None and datum["id"] not in subset:
                continue
            if i >= gen_k and subset is None:
                break

            first_pose = datum["initial"]
            seq_iter = model(text=datum["text"], first_pose=first_pose)
            seq_list = [seq for seq in seq_iter]
            stacked_tensor = torch.stack(seq_list, dim=0)

            if i == 0:
                device = next(model.parameters()).device
                original_pose = datum["pose"]["data"].to(device).squeeze()
                stacked_tensor = stacked_tensor.to(device).squeeze()

                if stacked_tensor.shape[0] > original_pose.shape[0]:
                    stacked_tensor = stacked_tensor[: original_pose.shape[0], :, :]
                else:
                    original_pose = original_pose[: stacked_tensor.shape[0], :, :]
                original_pose = torch.round(original_pose * 100) / 100
                stacked_tensor = torch.round(stacked_tensor * 100) / 100

                sq_error = torch.pow(original_pose - stacked_tensor, 2).sum(-1)

                print(f"Text: {datum['text']}")
                print("Original Pose:           Generated Pose:")
                print(f"{original_pose}    {stacked_tensor}")

                print("Squared Error:")
                num_elements = sq_error.numel()
                mse = sq_error.sum() / num_elements
                print(f"Mean Squared Error (MSE): {mse.item()}")
                difference_tensor = original_pose - stacked_tensor
                print("Difference Tensor:")
                print(difference_tensor)

                save_tensor_to_file(stacked_tensor, "generated.txt")
                save_tensor_to_file(original_pose, "adjusted.txt")

                visualize_3d_points(
                    original_pose.cpu().numpy(),
                    image_size=(1920, 1080),
                    focal_length=1.0,
                    center_x=660,
                    center_y=580,
                    scale_factor=800,
                    video_filename="target.mp4",
                )

                print("Visualizing Generated Pose:")
                visualize_3d_points(
                    stacked_tensor.cpu().numpy(),
                    image_size=(1920, 1080),
                    focal_length=2.0,
                    center_x=660,
                    center_y=580,
                    scale_factor=800,
                    video_filename="generated.mp4",
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
        raise Exception("Optimizer not supported. Use Adam or SGD.")


if __name__ == "__main__":
    # Assume args is a dictionary with the necessary parameters
    args = {
        "hidden_dim": 128,
        "text_encoder_depth": 2,
        "pose_encoder_depth": 4,
        "encoder_heads": 2,
        "max_seq_size": 200,
        "num_steps": 10,
        "tf_p": 0.5,
        "seq_len_weight": 2e-8,
        "noise_epsilon": 1e-4,
        "separate_positional_embedding": False,
        "encoder_dim_feedforward": 2048,
        "num_pose_projection_layers": 1,
        "optimizer": "Adam",
        "model_name": "test_model",
        "output_dir": "output",
    }

    model_args = get_model_args(args, 55, 3)
    model = TextGuidedPoseGenerationModel(**model_args)

    # Load the saved model weights
    model.load_state_dict(torch.load("pretrained_models/ckpt_Text2LIS_2024-07-16_1258/model_10step.ckpt", map_location=torch.device("cpu")), strict=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    train_dataset, test_dataset = get_dataset(num_samples=1, max_seq_size=200, split_ratio=0.9)
    pred(model, test_dataset, os.path.join(f"./models/{args['model_name']}", args["output_dir"], "train"))
