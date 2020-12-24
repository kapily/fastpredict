from typing import List
import torch
from fastai.vision.all import load_learner, ImageDataLoaders, Resize, Learner


def clear_splits(dls):
    for loader_idx, loader in enumerate(dls.loaders):
        for tls_idx, tls in enumerate(loader.dataset.tls):
            tls.splits.clear()
            dls.loaders[loader_idx].tls[tls_idx] = dls.loaders[loader_idx].tls[tls_idx].new_empty()


def clear_loaders(dls):
    dls.loaders.clear()


class FastPredict(object):

    @classmethod
    def clean_learner(cls, learn):
        clear_splits(learn.dls)
        clear_splits(learn.__stored_args__['dls'])
        # clear_loaders(learn.dls)
        clear_loaders(learn.__stored_args__['dls'])

    @classmethod
    def is_cuda_available(cls):
        return torch.cuda.is_available()

    @classmethod
    def get_device_str(cls):
        return 'cuda' if cls.is_cuda_available() else 'cpu'

    @classmethod
    def from_path(cls, learner_path, device='auto', expected_vocab=None):
        if device == 'cpu':
            cpu = True
        elif device == 'cuda':
            cpu = False
        else:
            assert device == 'auto'
            cpu = not cls.is_cuda_available()
            device = 'cpu' if cpu else 'cuda'
        learner = load_learner(learner_path, cpu=cpu)
        learner.model.to(device)
        if not cpu:
            assert device == 'cuda'
            is_cuda = next(learner.model.parameters()).is_cuda
            assert is_cuda
        return cls(learner, device, expected_vocab=expected_vocab)

    def __init__(self, learner, device, expected_vocab=None):
        # NOTE: we purposefully perform the imports inside the function. If we do it outside,
        # Python complains because the torch imports are getting forked
        # self.version = version
        self.learner = learner
        self.device = device
        self.vocab = self.learner.dls.vocab
        if self.device == 'cuda':
            self.dls = self.learner.dls.cuda()
        else:
            self.dls = self.learner.dls
        # TODO - need to support other vocab in the future
        # assert len(ocab) == 2
        if expected_vocab:
            assert set(self.vocab) == set(expected_vocab), f'Error vocab is actually: {str(self.vocab)}'
        # self.true_idx = vocab.o2i[True]

    def predict_paths(self, paths, target_class=None) -> List[float]:
        assert target_class is not None, 'For now, target_class is required. in future, all classes will be returned'
        # TODO ensure that it is in CUDA if it is available
        #print('device type: ', type(self.learner.dls.device))
        #print('device dir: ', dir(self.learner.dls.device))
        #print('device type2: ', self.learner.dls.device.type)

        assert self.dls.device.type == self.device
        dataloader = self.dls.test_dl(paths, device=self.device, num_workers=0)
        #if CUDA_AVAILABLE and 'cuda' not in self.learner.dls.device:
        #    logging.error('error: cuda not available for DLS')
        # self.learner.model.to('cuda')
        #print('learn.dls.device: ', self.learner.dls.device)
        #print('learner.model.parameters.is_cuda: ', next(self.learner.model.parameters()).is_cuda)
        #if self.device == 'cuda':
        #    dataloader.cuda()
        imgs, probs, classes, clas_idx = self.learner.get_preds(
            dl=dataloader, with_input=True, with_decoded=True)
        scores = [float(scores[self.vocab.o2i[target_class]]) for scores in probs]
        assert len(scores) == len(paths)
        return scores

    def predict_path(self, path, target_class=None):
        assert isinstance(path, str)
        scores = self.predict_paths([path], target_class=target_class)
        assert len(scores) == 1
        return scores[0]


class FastPredictBool(FastPredict):

    def predict_paths(self, paths, target_class=None) -> List[float]:
        if target_class is None:
            target_class = self.true_word
        return super().predict_paths(paths, target_class=target_class)

    def predict_path(self, path, target_class=None):
        if target_class is None:
            target_class = self.true_word
        return super().predict_path(path, target_class=target_class)

    def __init__(self, learner, device, expected_vocab=('false', 'true')):
        assert len(expected_vocab) == 2
        self.false_word = expected_vocab[0]
        self.true_word = expected_vocab[1]
        super().__init__(learner, device, expected_vocab=expected_vocab)

