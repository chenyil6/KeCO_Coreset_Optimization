from classification_utils import IMAGENET_1K_CLASS_ID_TO_LABEL
from torchvision.datasets import ImageFolder
from collections import defaultdict
import pickle
import os

class ImageNetDataset(ImageFolder):
    """Class to represent the ImageNet1k dataset."""
    
    def __init__(self, root, index_file=None, **kwargs):
        super().__init__(root=root, **kwargs)
        
        # 尝试从文件加载索引，如果没有文件则构建索引
        if index_file and self.load_class_index(index_file):
            print(f"Loaded class indices from {index_file}")
        else:
            self.class_to_indices = self.build_class_index()
            if index_file:
                self.save_class_index(index_file)
    
    def build_class_index(self):
        """Build a dictionary mapping each class to the indices of its images."""
        class_to_indices = defaultdict(list)
        for idx, (_, class_id) in enumerate(self.imgs):
            class_to_indices[class_id].append(idx)
        return class_to_indices

    def save_class_index(self, output_file):
        """Save the class indices to a file for faster loading next time."""
        with open(output_file, 'wb') as f:
            pickle.dump(self.class_to_indices, f)
        print(f"Saved class indices to {output_file}")

    def load_class_index(self, input_file):
        """Load the class indices from a file."""
        try:
            with open(input_file, 'rb') as f:
                self.class_to_indices = pickle.load(f)
            return True
        except Exception as e:
            print(f"Failed to load class indices from {input_file}: {e}")
            return False
    
    def __getitem__(self, idx):
        sample, target,path = super().__getitem__(idx)
        target_label = IMAGENET_1K_CLASS_ID_TO_LABEL[target]
        return {
            "id": idx,
            "image": sample,
            "image_path":path,
            "class_id": target,  # numeric ID of the ImageNet class
            "class_name": target_label,  # human-readable name of ImageNet class
        }
    
    def get_data_list_by_class(self, class_id=None, class_name=None,max_samples = 150):
        """Get the list of image data for a given class."""
        if class_id is not None:
            indices = self.class_to_indices[class_id]
        elif class_name is not None:
   
            class_id = {v: k for k, v in IMAGENET_1K_CLASS_ID_TO_LABEL.items()}.get(class_name)
            if class_id is None:
                return []
            indices = self.class_to_indices[class_id]
        else:
            return []
        
    
        indices = indices[:max_samples]

    
        return [self.__getitem__(idx) for idx in indices]