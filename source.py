from collections import Counter

import torch.nn
from oml import const
from oml.interfaces.datasets import IDatasetWithLabels
from oml.interfaces.models import IExtractor
from oml.interfaces.models import IPairwiseModel


class BioDatasetWithLabels(IDatasetWithLabels):

    def __init__(self, labels, categories, is_first_type, descriptors):
        assert all(c == 2 for _, c in Counter(labels).items())

        self.labels = labels
        self.categories = categories
        self.is_first_type = is_first_type
        self.descriptors = descriptors

    def __getitem__(self, idx):
        item = {
            const.INPUT_TENSORS_KEY: self.descriptors[idx],
            const.LABELS_KEY: self.labels[idx],
            const.CATEGORIES_KEY: self.categories[idx],
            "is_first_type": self.is_first_type[idx]
        }
        return item

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels


class SimpleExtractor(IExtractor):

    def __init__(self, in_dim, out_dim):
        super(SimpleExtractor, self).__init__()
        self.fc = torch.nn.Linear(in_features=in_dim, out_features=out_dim)

    def forward(self, x):
        return self.fc(x)

    def feat_dim(self):
        return self.fc.out_features


class SimpleSiamese(IPairwiseModel):

    def __init__(self, extractor1: IExtractor, extractor2: IExtractor):
        super(SimpleSiamese, self).__init__()
        self.extractor1 = extractor1
        self.extractor2 = extractor2

        self.head = torch.nn.Linear(
            in_features=self.extractor1.feat_dim + self.extractor2.feat_dim,
            out_features=1
        )

    def forward(self, x1, x2):
        x1 = self.extractor1(x1)
        x2 = self.extractor2(x2)
        x = self.head(torch.concat([x1, x2]))
        return x


class PairsSamplerTwoModalities:

    def __init__(self):
        super(PairsSamplerTwoModalities, self).__init__()

    def sample(self, features_a, features_b, labels_a, labels_b):
        pass


features_a = torch.randn((4, 10))
features_b = torch.randn((4, 10))
labels_a = torch.tensor([0, 1, 2, 3])
labels_b = torch.tensor([0, 1, 2, 3])
