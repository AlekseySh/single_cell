from collections import Counter
from pprint import pprint

import torch.nn
from oml import const
from oml.interfaces.datasets import IDatasetWithLabels, IDatasetQueryGallery
from oml.interfaces.models import IExtractor
from oml.interfaces.models import IPairwiseModel
from oml.utils.misc_torch import pairwise_dist
from torchvision.ops import MLP


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
            "is_first_type": bool(self.is_first_type[idx])
        }
        return item

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.labels


class BioDatasetQueryGallery(IDatasetQueryGallery):

    def __init__(self, labels, is_first_type, descriptors):
        assert all(c == 2 for _, c in Counter(labels).items())

        self.labels = labels
        self.is_first_type = is_first_type
        self.descriptors = descriptors

    def __getitem__(self, idx):
        # assume 1st type of records are always queries
        is_query = bool(self.is_first_type[idx])

        item = {
            const.LABELS_KEY: self.labels[idx],
            const.EMBEDDINGS_KEY: self.descriptors[idx],
            const.IS_QUERY_KEY: is_query,
            const.IS_GALLERY_KEY: not is_query
        }
        return item

    def __len__(self):
        return len(self.labels)


class SimpleExtractor(IExtractor):

    def __init__(self, in_dim, out_dim):
        super(SimpleExtractor, self).__init__()
        self.out_dim = out_dim
        self.head = MLP(in_channels=in_dim, hidden_channels=[32, 32, 32, out_dim])

    def forward(self, x):
        return self.head(x)

    @property
    def feat_dim(self):
        return self.out_dim


class SimpleSiamese(IPairwiseModel):

    def __init__(self, extractor1: IExtractor, extractor2: IExtractor):
        super(SimpleSiamese, self).__init__()
        self.extractor1 = extractor1
        self.extractor2 = extractor2

        self.head = MLP(
            #in_channels=self.extractor1.embedding_dim + self.extractor2.embedding_dim,
            in_channels = 64*2,
            hidden_channels=[32, 32, 1]
        )

    def forward(self, x1, x2):
        x1 = self.extractor1(x1)
        x2 = self.extractor2(x2)
        x = torch.concat([x1, x2], dim=-1)
        x = self.head(x)
        x = x.view(len(x))
        return x
    
    def predict(self, x1, x2):
        x = self.forward(x1,x2)
        return x


class PairsSamplerTwoModalities:

    def __init__(self, hard: bool):
        super(PairsSamplerTwoModalities, self).__init__()
        self.hard = hard

    def sample(self, features_a, features_b, labels_a, labels_b):
        if self.hard:
            return self.sample_hard(features_a, features_b, labels_a, labels_b)
        else:
            return self.sample_all(features_a, features_b, labels_a, labels_b)

    def sample_hard(self, features_a, features_b, labels_a, labels_b):
        # Note, the code was not tested properly
        assert (labels_a.unique() == labels_b.unique()).all(), (labels_a, labels_b)
        assert (len(features_a) == len(labels_a)) and (len(features_b) == len(labels_b))

        n = len(features_a)

        distances = pairwise_dist(features_a, features_b)
        is_positive = labels_a.unsqueeze(-1) == labels_b
        assert distances.shape == is_positive.shape

        distances_pos = distances.clone()
        distances_pos[~is_positive] = -float("inf")
        ii_hard_pos = distances_pos.argmax(dim=0)

        distances_neg = distances.clone()
        distances_neg[is_positive] = +float("inf")
        ii_hard_neg = distances_neg.argmin(dim=0)

        ii = torch.arange(n)

        ids_1 = torch.concat([ii, ii], dim=0)
        ids_2 = torch.concat([ii_hard_pos, ii_hard_neg], dim=0)

        is_negative = torch.ones(2 * n).bool()
        is_negative[:n] = False

        return ids_1, ids_2, is_negative

    def sample_all(self, features_a, features_b, labels_a, labels_b):
        assert (labels_a.unique() == labels_b.unique()).all(), (labels_a, labels_b)
        assert (len(features_a) == len(labels_a)) and (len(features_b) == len(labels_b))

        ids1 = []
        ids2 = []
        is_negative = []

        for i in range(len(labels_a)):
            for j in range(len(labels_b)):
                ids1.append(i)
                ids2.append(j)
                is_negative.append(labels_a[i] != labels_b[j])

        return torch.tensor(ids1).long(), torch.tensor(ids2).long(), torch.tensor(is_negative).bool()


def check_miner():
    labels_a = torch.tensor([0, 1, 2, 3])
    labels_b = torch.tensor([1, 2, 3, 0, 0])

    features_a = torch.ones((4, 3)) * labels_a.view(-1, 1)
    features_b = torch.ones((5, 3)) * labels_b.view(-1, 1) + 0.1

    ids_1, ids_2, is_negative = PairsSamplerTwoModalities(hard=False).sample(
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
