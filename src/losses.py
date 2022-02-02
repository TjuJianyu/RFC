import torch.nn as nn
from wilds.common.metrics.loss import ElementwiseLoss, Loss, MultiTaskLoss
from wilds.common.metrics.all_metrics import MSE
from utils import cross_entropy_with_logits_loss
from wilds.common.metrics.metric import ElementwiseMetric
from wilds.common.utils import  maximum, minimum
class RFCmtAccuracy(ElementwiseMetric):
    def __init__(self, prediction_fn=None, name=None):
        self.prediction_fn = prediction_fn
        if name is None:
            name = 'acc'
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred, y_true):
        #print('hh',self.prediction_fn)
        if self.prediction_fn is not None:
            assert y_pred.shape[1] % y_true.shape[1] == 0
            acc = 0
            d_out = y_pred.shape[1] // y_true.shape[1]
            
            for i in range(y_true.shape[1]):
                per_y_pred = self.prediction_fn(y_pred[:,i*d_out : (i+1)*d_out])
                per_y_true  = y_true[:,i]
                per_acc = (per_y_pred == per_y_true).float()
                acc += per_acc

            return acc / y_true.shape[1]
        else:
            
            raise NotImplementedError    

    def worst(self, metrics):
        return minimum(metrics)

class RFCmtElementwiseLoss(ElementwiseMetric):
    def __init__(self, loss_fn, name=None):
        self.loss_fn = loss_fn
        if name is None:
            name = 'loss'
        super().__init__(name=name)

    def _compute_element_wise(self, y_pred, y_true):
        """
        Helper for computing element-wise metric, implemented for each metric
        Args:
            - y_pred (Tensor): Predicted targets or model output
            - y_true (Tensor): True targets
        Output:
            - element_wise_metrics (Tensor): tensor of size (batch_size, )
        """
        #print(y_pred.shape, y_true.shape)
        assert y_pred.shape[1] % y_true.shape[1] == 0

        loss = 0
        d_out = y_pred.shape[1] // y_true.shape[1]
        for i in range(y_true.shape[1]):
            loss += self.loss_fn(y_pred[:,i*d_out : (i+1)*d_out], y_true[:,i])
        return loss 


    def worst(self, metrics):
        """
        Given a list/numpy array/Tensor of metrics, computes the worst-case metric
        Args:
            - metrics (Tensor, numpy array, or list): Metrics
        Output:
            - worst_metric (float): Worst-case metric
        """
        return maximum(metrics)

# class RFCmtThrElementwiseLoss(ElementwiseMetric):
#     def __init__(self, loss_fn, name=None):
#         self.loss_fn = loss_fn
#         if name is None:
#             name = 'loss'
#         super().__init__(name=name)

#     def _compute_element_wise(self, y_pred, y_true):
#         """
#         Helper for computing element-wise metric, implemented for each metric
#         Args:
#             - y_pred (Tensor): Predicted targets or model output
#             - y_true (Tensor): True targets
#         Output:
#             - element_wise_metrics (Tensor): tensor of size (batch_size, )
#         """
#         #print(y_pred.shape, y_true.shape)
#         assert y_pred.shape[1] % y_true.shape[1] == 0

#         loss = 0
#         d_out = y_pred.shape[1] // y_true.shape[1]
#         for i in range(y_true.shape[1]):
#             loss += self.loss_fn(y_pred[:,i*d_out : (i+1)*d_out], y_true[:,i])
#         return loss 


#     def worst(self, metrics):
#         """
#         Given a list/numpy array/Tensor of metrics, computes the worst-case metric
#         Args:
#             - metrics (Tensor, numpy array, or list): Metrics
#         Output:
#             - worst_metric (float): Worst-case metric
#         """
#         return maximum(metrics)


def initialize_loss(loss, config):
    if loss == 'cross_entropy':
        return ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))
    elif loss == 'rfcmt_cross_entropy':
        return RFCmtElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))
    elif loss == 'lm_cross_entropy':
        return MultiTaskLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))

    elif loss == 'mse':
        return MSE(name='loss')

    elif loss == 'multitask_bce':
        return MultiTaskLoss(loss_fn=nn.BCEWithLogitsLoss(reduction='none'))

    elif loss == 'fasterrcnn_criterion':
        from models.detection.fasterrcnn import FasterRCNNLoss
        return ElementwiseLoss(loss_fn=FasterRCNNLoss(config.device))

    elif loss == 'cross_entropy_logits':
        return ElementwiseLoss(loss_fn=cross_entropy_with_logits_loss)

    else:
        raise ValueError(f'loss {loss} not recognized')
