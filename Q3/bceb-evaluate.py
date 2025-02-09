import os
import sys

from typing import Iterator
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Redfine classes in case you changed Them

class TestNode:
    def __init__(self, name):
        self.name = name
        self._count = 0
        self.children = {}
        self._entities = []

    def add_to_node(self, path, entity, level=0):
        if level >= len(path):
            self._entities.append(entity)
            return
        part = path[level]
        if part not in self.children:
            self.children[part] = TestNode(path[:level+1])
        self.children[part].add_to_node(path, entity, level=level+1)
        self._count += 1

    @property
    def is_leaf(self):
        return len(self._entities) > 0

    @property
    def count(self):
        if self.is_leaf:
            return len(self._entities)
        else:
            return self._count

    @property
    def entities(self):
        if self.is_leaf:
            return list((entity, self.name) for entity in self._entities)
        else:
            child_entities = []
            for child in self.children.values():
                child_entities.extend(child.entities)
        return child_entities

    def level_iterator(self, level=None):
        """
        iterates a certain depth in a tree and returns the nodes
        """
        if level == 0:
            yield self
        elif level == None and self.is_leaf:
            yield self
        elif self.is_leaf and level != 0:
            raise Exception("Incorrect level is specified in tree.")
        else:
            if level is not None:
                level -= 1
            for child in self.children.values():
                for v in child.level_iterator(level):
                    yield v


    def print_node(self, level=0, max_level=None):
        leaves = 1
        print(' ' * (level * 4) + f"{self.name[-1]} ({self.count})")
        for node in self.children.values():
            if max_level is None or level < max_level:
                leaves += node.print_node(level + 1, max_level=max_level)
        return leaves

class TestHiererchicalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, level=None):
        self.tree = TestNode("Dataset") # keeps the group information of self.data in a tree (per index).
        self.level = level
        if level is None:
            self.level = 7  # Hardcoded
        self.classes = set()
        data = []
        index = 0
        for group_name in sorted(os.listdir(dataset_path)):
            if not os.path.isdir(os.path.join(dataset_path, group_name)):
                continue
            for image_name in sorted(os.listdir(os.path.join(dataset_path, group_name))):
                group = tuple(group_name.split("_")[1:])
                image_path = os.path.join(dataset_path, group_name, image_name)
                data.append({
                        "image_path": image_path,
                        "group": group,
                    }
                )
                self.tree.add_to_node(group, index)
                index += 1
                self.classes.add(group[:self.level])
        self.data = data
        self.classes = {group: index for (index, group) in enumerate(sorted(list(self.classes)))}
        self.secret_transform = nn.Identity()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(232),
            transforms.CenterCrop(224),
            self.secret_transform,
            transforms.Normalize((0.4556, 0.4714, 0.3700), (0.2370, 0.2318, 0.2431)),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx]["image_path"])
        target = self.classes[self.data[idx]["group"][:self.level]]
        if self.transform:
            image = self.transform(image)

        return image, target

    def get_group_iterator(self, level=None) -> Iterator[TestNode]:
        for group in self.tree.level_iterator(level):
            yield group



def evaluate_group(results, indices, device):
    indices = torch.tensor(indices, device=device)
    results = results[indices[0]:indices[-1]+1]
    acc = sum(results).float().cpu().item() / len(indices)

    return acc

def evaluate_model(model, ds: TestHiererchicalDataset, level, k, device, batch_size=1024):
    test_loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    
    model.eval()
    results = torch.zeros(len(ds), device=device)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(tqdm(test_loader)):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            st = batch_idx*batch_size
            results[st:st+len(labels)] = (predicted == labels)


    group_results = []
    for group in tqdm(list(ds.get_group_iterator(level=level))):
        group_indices = np.array(list(index for (index, _) in group.entities)).astype(int)

        group_results.append(evaluate_group(results=results, indices=group_indices, device=device))

    return torch.mean(torch.topk(torch.tensor(group_results), k=k, largest=False, sorted=True).values).cpu().item()
    
# How to use:
# evaluate_model(model, ds, level=7, k=K, device=device)
