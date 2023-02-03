from collections import Counter
from pprint import pprint

import torch.nn
from oml import const
from oml.interfaces.datasets import IDatasetWithLabels
from oml.interfaces.models import IExtractor
from oml.interfaces.models import IPairwiseModel
from oml.utils.misc_torch import elementwise_dist, assign_2d, pairwise_dist


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

    @property
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
        x = torch.concat([x1, x2], dim=-1)
        x = self.head(x)
        x = x.view(len(x))
        return x


class PairsSamplerTwoModalities:

    def __init__(self):
        super(PairsSamplerTwoModalities, self).__init__()

    def sample(self, features_a, features_b, labels_a, labels_b):
        n = len(features_a)

        distances = pairwise_dist(features_a, features_b)
        is_positive = labels_a.unsqueeze(-1) == labels_b

        distances_pos = distances.clone()
        distances_pos[~is_positive] = -float("inf")
        ii_hard_pos = distances_pos.argmax(dim=1)

        distances_neg = distances.clone()
        distances_neg[is_positive] = +float("inf")
        ii_hard_neg = distances_neg.argmin(dim=1)

        ii = torch.arange(n)

        ids_1 = torch.concat([ii, ii], dim=0)
        ids_2 = torch.concat([ii_hard_pos, ii_hard_neg], dim=0)

        is_negative = torch.ones(2 * n).bool()
        is_negative[:n] = False

        return ids_1, ids_2, is_negative


def check_miner():
    labels_a = torch.tensor([0, 1, 2, 3])
    labels_b = torch.tensor([1, 2, 3, 0])

    features_a = torch.ones((4, 3)) * labels_a.view(-1, 1)
    features_b = torch.ones((4, 3)) * labels_b.view(-1, 1) + 0.1

    ids_1, ids_2, is_negative = PairsSamplerTwoModalities().sample(
        features_a,
        features_b,
        labels_a,
        labels_b
    )

    pprint(list(zip(
        features_a[ids_1],
        features_b[ids_2],
        is_negative
    )))


if __name__ == '__main__':
    check_miner()
