import os
import pandas as pd
import torch
import numpy as np
from typing import List, Dict, Tuple, Any
from pose_format import Pose, PoseHeader, PoseBody
from pose_format.pose_header import PoseHeaderComponent, PoseHeaderDimensions
from tqdm import tqdm
from transformers import CLIPTokenizer

from text2lis.data.dataset import Text2MotionDataset
from text2lis.model.colator import zero_pad_collator  # Assicurati che questa importazione funzioni nel tuo contesto
from torch.utils.data import Dataset, DataLoader

MIN_CONFIDENCE = 0.2
NUM_HAND_KEYPOINTS = 22
NUM_FACE_KEYPOINTS = 70


# Definizione della classe per i dati sintetici
class TextPoseDatum(Dict):
    id: str
    text: str
    pose: Pose
    length: int
    initial: [[]]


class TextPoseDataset(Dataset):
    def __init__(self, data: List[TextPoseDatum]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        datum = self.data[index]
        if "pose" not in datum:
            return datum

        pose = datum["pose"]
        initial_pose = datum["initial"]
        torch_body = pose.body.data  # Data in `Pose` potrebbe essere `body_data`
        pose_length = len(torch_body)

        return {
            "id": datum["id"],
            "text": datum["text"],
            "pose": {
                "obj": pose,
                "data": torch_body,
                "confidence": pose.body.confidence,
                "length": torch.tensor([pose_length], dtype=torch.float),
                "inverse_mask": torch.ones(pose_length, dtype=torch.bool),

            },
            "initial": torch.tensor(initial_pose, dtype=torch.float)
        }


def load_keypoints(keypoint_file_path):
    if os.path.exists(keypoint_file_path):
        df_keypoints = pd.read_excel(keypoint_file_path, sheet_name='Sheet1')

        valori_numerici = []
        valori_numerici2 = []

        for index, row in df_keypoints.iterrows():
            stringa_dizionari = row[2]
            stringa_ = row[1]

            lista_dizionari = eval(stringa_dizionari)
            lista = eval(stringa_)

            for dizionario in lista_dizionari:
                for punto in dizionario:
                    valori_numerici.append((punto['x'], punto['y'], punto['z']))

            for dizionario in lista:
                for punto in dizionario:
                    valori_numerici.append((punto['x'], punto['y'], punto['z']))
        keypoints_matrix = np.array(valori_numerici)

        return keypoints_matrix


# Funzione per creare un dato sintetico di pose


def load_initial(output_file):
    loaded_values = []
    with open(output_file, "r") as f:
        for line in f:
            loaded_values.append(float(line.strip()))
    return loaded_values


def generate_pose_datum(id: str, text: str, pose_length: int, pose: List) -> TextPoseDatum:
    num_keypoints = 55
    pose_dim = 3  # 3D space: (x, y, confidence)

    # Percorso del file Excel
    file = r"text2lis/model/initial_pose.txt"

    # Crea la matrice dei keypoints
    keypoints_matrix = load_initial(file)

    # Generazione dei dati di pose sintetici
    data = torch.stack(pose, dim=0)  # Dimensioni: [seq_len, num_keypoints, pose_dim]
    confidence = torch.full((pose_length, num_keypoints), 1)  # Dimensioni: [seq_len, num_keypoints]

    # Creazione delle dimensioni dell'header
    width, height, depth = pose_length, num_keypoints, pose_dim
    dimensions = PoseHeaderDimensions(width=width, height=height, depth=depth)

    # Creazione dei componenti dell'header
    components = []
    for i in range(num_keypoints):
        component = PoseHeaderComponent(
            name=f"keypoint_{i}",
            points=[f"point_{i}"],
            limbs=[(i, i)] if i > 0 else [],
            colors=[(0, 0, 0)],  # Colore fittizio
            point_format="x,y,confidence"
        )
        components.append(component)

    # Creazione dell'header e del body per il pose
    header = PoseHeader(version=1.0, dimensions=dimensions, components=components, is_bbox=False)
    body = PoseBody(data=data.unsqueeze(1), confidence=confidence.unsqueeze(1), fps=30)  # Aggiungi una dimensione batch

    pose = Pose(header=header, body=body)

    return TextPoseDatum({
        "id": id,
        "text": text,
        "pose": pose,
        "length": len(pose.body.data),
        "initial": keypoints_matrix
    })


# Funzione per processare un dato sintetico in un formato compatibile con TextPoseDatum
def process_synthetic_datum(datum: TextPoseDatum) -> TextPoseDatum:
    text = datum["text"]
    pose = datum["pose"]
    pose_length = datum["length"]
    initial = datum["initial"]

    face_th = 0.5 * NUM_FACE_KEYPOINTS
    hands_th = MIN_CONFIDENCE

    # Prune all leading frames containing only zeros, almost no face or no hands
    for i in range(len(pose.body.data)):
        if pose.body.confidence[i, :, 25:-42].sum() > face_th and \
                pose.body.confidence[i, :, 4] + pose.body.confidence[i, :, 7] > hands_th:
            if i != 0:
                pose.body.data = pose.body.data[i:]
                pose.body.confidence = pose.body.confidence[i:]
            break

    # Prune all trailing frames containing only zeros, almost no face or no hands
    for i in range(len(pose.body.data) - 1, 0, -1):
        if pose.body.confidence[i, :, 25:-42].sum() > face_th and \
                pose.body.confidence[i, :, 4] + pose.body.confidence[i, :, 7] > hands_th:
            if i != len(pose.body.data) - 1:
                pose.body.data = pose.body.data[:i + 1]
                pose.body.confidence = pose.body.confidence[:i + 1]
            break

    return TextPoseDatum({
        "id": datum["id"],
        "text": text,
        "pose": pose,
        "length": len(pose.body.data),
        "initial": initial
    })


# Funzione per creare un dataset di dati sintetici
def create_synthetic_dataset(num_samples: int = 1000, max_seq_size: int = 200) -> List[TextPoseDatum]:
    dataset = []
    phrases = [
        "How are you?",
        "Nice to meet you",
        "Thank you for your exceptional and thoughtful help.",
        "Congratulations!",
        "I'm lost",

    ]
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    dataset_ = Text2MotionDataset(r"LIS", tokenizer)
    for idx in tqdm(range(len(dataset_))):
        if idx!= 5099:


            text, gesture_list, length = dataset_[idx]
            id = f"sample_{idx}"
            pose_length = length
            datum = generate_pose_datum(id, text, pose_length, gesture_list)
            dataset.append(datum)


    return dataset


"""
    #print(len(phrases))
    for i in range(len(phrases)):
        id = f"sample_{i}"
        # Testo sintetico
        pose_length = np.random.randint(100, 105)  # Lunghezza della sequenza sintetica
        datum = 
        dataset.append(datum)
    return dataset
    """


# Funzione per creare il dataset sintetico e suddividerlo in train e test
def get_dataset(num_samples: int = 1000, max_seq_size: int = 200, split_ratio: float = 0.9) -> Tuple[TextPoseDataset, TextPoseDataset]:
    synthetic_data = create_synthetic_dataset(num_samples=num_samples, max_seq_size=max_seq_size)
    processed_data = [process_synthetic_datum(datum) for datum in synthetic_data]

    split_ratio = 0.8  # 80% dei dati per il training e 20% per il testing
    split_index = int(len(processed_data) * split_ratio)

    # Divide i dati in train e test
    train_data = processed_data[:split_index]
    test_data = processed_data[split_index:]

    train_dataset = TextPoseDataset(train_data)
    test_dataset = TextPoseDataset(test_data)

    return train_dataset, test_dataset


if __name__ == '__main__':
    train_dataset, test_dataset = get_dataset(num_samples=1000, max_seq_size=200, split_ratio=0.9)

    # Verifica dei dati
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=zero_pad_collator)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=zero_pad_collator)

    for batch in train_loader:
        print(batch)

