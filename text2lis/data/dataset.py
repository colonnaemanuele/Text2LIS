import os
from os.path import join as pjoin
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTokenizer

# from logger import logger
import pandas as pd
import pickle


class Text2MotionDataset(Dataset):
    def __init__(self, data_root, tokenizer):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.data_dict = {}
        self.video_paths = os.listdir(pjoin(self.data_root, "integration"))
        video_list = self._get_videos()
        xslx_dict = self._get_xlsx()

        for index, video in enumerate(video_list):
            filename = video.split(".")[0]
            self.data_dict[index] = {
                "video": video,
                "text": xslx_dict[filename],
                "motion": f"{filename}.pkl",
            }

    def _get_videos(self):
        video_list = []
        for video in self.video_paths:
            video_path = pjoin(self.data_root, "video_tagliati", video)
            video_list.extend(os.listdir(video_path))
        return video_list

    def _get_xlsx(self) -> dict:
        xlsx_dict = {}
        for video in self.video_paths:
            dataframe = pd.read_excel(
                pjoin(self.data_root, "xlsx", f"{video}.xlsx"), header=None
            )
            for index, row in dataframe.iterrows():
                xlsx_dict[f"{video}_{index}"] = row[2]
        return xlsx_dict

    def __len__(self):
        return len(self.data_dict)

    def _get_motion_list(self, motion_path):
        with open(motion_path, "rb") as f:
            motion_dict = pickle.load(f)
            return motion_dict["joints"]["body"]

    def __getitem__(self, idx):
        data = self.data_dict[idx]
        motion_path = pjoin(
            self.data_root,
            "integration",
            data["motion"].rsplit("_", 1)[0],
            data["motion"],
        )
        print(motion_path)
        gesture_list = self._get_motion_list(motion_path)
        text = self.tokenizer(
            data["text"], return_tensors="pt", padding=True, truncation=True
        )
        text = text["input_ids"].squeeze(0)
        # text_attention_mask = text["attention_mask"].squeeze(0)

        return text, gesture_list, #text_attention_mask


if __name__ == "__main__":
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    dataset = Text2MotionDataset("LIS/LISDataset", tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(next(iter(dataloader)))
