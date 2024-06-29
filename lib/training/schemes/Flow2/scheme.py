import torch
import torch.nn.functional as F
import numpy as np

from lib.training.training import cached_property
from ..digt_mol_training import DIGT_MOL_Training

from lib.models.Flow2 import DIGT_Flow2
from lib.data.Flow2 import Flow2StructuralPEGraphDataset


class Flow2_Training(DIGT_MOL_Training):
    def get_default_config(self):
        config_dict = super().get_default_config()
        config_dict.update(
            dataset_name='Flow2',
            dataset_path='cache_data/Flow2',
            evaluation_type='prediction',
            predict_on=['test'],
            state_file=None,
        )
        return config_dict

    def get_dataset_config(self):
        dataset_config, _ = super().get_dataset_config()
        return dataset_config, Flow2StructuralPEGraphDataset

    def get_model_config(self):
        model_config, _ = super().get_model_config()
        return model_config, DIGT_Flow2

    def calculate_ce_loss(self, outputs, targets):

        # targets = targets.type(torch.LongTensor)
        return F.cross_entropy(outputs, targets)

    def calculate_loss(self, outputs, inputs):
        return self.calculate_ce_loss(outputs, inputs['target'])

    '''
    @cached_property
    def evaluator(self):
        from ogb.graphproppred import Evaluator
        evaluator = Evaluator(name = "ogbg-molhiv")
        return evaluator
    '''

    def prediction_step(self, batch):
        return dict(
            predictions=torch.softmax(self.model(batch), dim=1),
            targets=batch['target'],
        )

    def evaluate_predictions(self, predictions):
        y_true = predictions['targets']
        y_pred = predictions['predictions']
        y_pred = np.argmax(y_pred, axis=1)
        acc = np.sum(y_true == y_pred) / np.sum(len(y_true))

        results = {'acc': acc}
        for k, v in results.items():
            if hasattr(v, 'tolist'):
                results[k] = v.tolist()

        return results

    def evaluate_on(self, dataset_name, dataset, predictions):
        print(f'Evaluating on {dataset_name}')
        results = self.evaluate_predictions(predictions)
        return results


SCHEME = Flow2_Training
