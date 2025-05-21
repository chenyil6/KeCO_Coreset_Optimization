from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

from classification_utils import *
import pickle
import os
import scipy.io
from PIL import Image

class StanfordDogDataset(Dataset):
    """Class to represent the Stanford Dogs dataset."""

    def __init__(self, root, transform=None, train=True,index_file=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.image_paths = []
        self.labels_ = []
        self.labels = []
        self.split_file = "train_list.mat" if self.train else "test_list.mat"
        self.class_id_to_name = dict(zip(range(len(STANFORD_DOG_CLASSNAMES)), STANFORD_DOG_CLASSNAMES))

        self.class_name2id = dict(
            zip(STANFORD_DOG_CLASSNAMES, range(len(STANFORD_DOG_CLASSNAMES)))
        )
        file_list = scipy.io.loadmat(os.path.join(self.root, self.split_file))['file_list']
        for item in file_list:
            file_path = item[0][0]
            self.image_paths.append(os.path.join(root, "Images", file_path))
            class_name = file_path.split("/")[0][10:]
            self.labels_.append(class_name)
            self.labels.append(self.class_name2id[class_name])
        
        if self.train:  # 只需要对训练集构建 class_to_indices
            # 尝试加载 class_to_indices.pkl，如果文件存在就直接加载，否则计算并保存
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


    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path, target_label = self.image_paths[idx], self.labels_[idx]
        sample = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        class_id = self.class_name2id[target_label]
        return {
            "id": idx,
            "image": sample,
            "class_id": class_id,
            "class_name": target_label,
            "image_path": image_path,
        }

    def _build_class_to_indices(self):
        """构建 class_id 到样本索引的映射"""
        class_to_indices = {class_id: [] for class_id in range(len(STANFORD_DOG_CLASSNAMES))}
        for idx, label in enumerate(self.labels):
            class_to_indices[label].append(idx)
        return class_to_indices
    
    def get_data_list_by_class(self, class_id):
        """
        获取某个类别的所有样本（图片路径 + 类别）。
        :param class_id: 目标类别索引 (0-119)
        :return: List[Dict]，包含所有该类别的样本
        """
        return [self.__getitem__(idx) for idx in self.class_to_indices[class_id]]
    