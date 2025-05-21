from torch.utils.data import Dataset

from classification_utils import *
import pickle
import os
from PIL import Image

class CUB200Dataset(Dataset):
    """Class to represent the CUB-200 dataset."""

    def __init__(self, root, index_file=None,transform=None, train=True):
        self.root = root
        self.transform = transform
        self.train = train
        self.image_paths = []
        self.labels = []
        self.class_id_to_name = dict(zip(range(len(CUB_CLASSNAMES)), CUB_CLASSNAMES))

        image_names = self._get_image_names()

        label_file = os.path.join(root, 'image_class_labels.txt')
        with open(label_file, 'r') as f:
            for line in f:
                image_id, class_id = line.strip().split()
                self.image_paths.append(os.path.join(root, 'images', image_names[int(image_id) - 1]))
                self.labels.append(int(class_id) - 1)  # Class IDs are 1-indexed

        split_file = os.path.join(root, 'train_test_split.txt')
        with open(split_file, 'r') as f:
            split_lines = f.readlines()

        if self.train:
            self.image_paths = [path for i, path in enumerate(self.image_paths) if int(split_lines[i].strip().split()[1]) == 1]
            self.labels = [label for i, label in enumerate(self.labels) if int(split_lines[i].strip().split()[1]) == 1]
        else:
            self.image_paths = [path for i, path in enumerate(self.image_paths) if int(split_lines[i].strip().split()[1]) == 0]
            self.labels = [label for i, label in enumerate(self.labels) if int(split_lines[i].strip().split()[1]) == 0]

        if self.train: 
            self.class_to_indices_path = index_file
            if os.path.exists(self.class_to_indices_path):
                with open(self.class_to_indices_path, 'rb') as f:
                    self.class_to_indices = pickle.load(f)
                print(f"Loaded class_to_indices from {self.class_to_indices_path}")
            else:
                self.class_to_indices = self._build_class_to_indices()
                with open(self.class_to_indices_path, 'wb') as f:
                    pickle.dump(self.class_to_indices, f)
                print(f"Saved class_to_indices to {self.class_to_indices_path}")

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        class_name = CUB_CLASSNAMES[label]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return {
            "id": idx,
            "image": img,
            "class_id": label,
            "class_name": class_name,
            "image_path": img_path,
        }
    
    def _get_image_names(self):
        image_names = []
        with open(os.path.join(self.root, 'images.txt'), 'r') as f:
            for line in f:
                _, image_name = line.strip().split()
                image_names.append(image_name)
        return image_names
    
    def __len__(self) -> int:
        return len(self.image_paths)

    def _build_class_to_indices(self):
        class_to_indices = {class_id: [] for class_id in range(len(CUB_CLASSNAMES))}
        for idx, label in enumerate(self.labels):
            class_to_indices[label].append(idx)
        return class_to_indices
    
    def get_data_list_by_class(self, class_id):
        return [self.__getitem__(idx) for idx in self.class_to_indices[class_id]]