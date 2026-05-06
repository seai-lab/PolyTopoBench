from pathlib import Path
from skimage import io
from skimage.transform import resize
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class CocoInferenceDataset(Dataset):
    """Generic COCO-style dataset used for PolyWorld inference."""

    def __init__(self, images_directory, annotations_path, window_size=320):

        self.images_directory = Path(images_directory)
        self.annotations_path = Path(annotations_path)

        self.coco = COCO(str(self.annotations_path))

        # Inference must cover every image, including empty tiles with no annotations.
        self.image_ids = sorted(self.coco.getImgIds())

        self.len = len(self.image_ids)

        self.window_size = window_size
        self.max_points = 256


    def load_sample(self, idx):

        idx = self.image_ids[idx]

        img = self.coco.loadImgs(idx)[0]
        image_path = self.images_directory / img["file_name"]

        image = io.imread(str(image_path))
        original_height, original_width = image.shape[:2]
        image = resize(image, (self.window_size, self.window_size, 3), anti_aliasing=True, preserve_range=True)

        image_idx = torch.tensor([idx])
        image = torch.from_numpy(image)
        image = image.permute(2,0,1) / 255.0

        sample = {
            "image": image,
            "image_idx": image_idx,
            "original_size": torch.tensor([original_height, original_width], dtype=torch.float32),
        }
        return sample


    def __len__(self):
        return self.len


    def __getitem__(self, idx):

        sample = self.load_sample(idx)
        return sample

