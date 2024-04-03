import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_EPS = 1e-10
PADDED_Y_VALUE = -1


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels)
    

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(labels == 1, probs, 1 - probs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# pearson correlation
def pearson_corr(x, y):
    vx = x - x.mean()
    vy = y - y.mean()
    cost = (vx * vy).sum() / ((vx ** 2).sum().sqrt() * (vy ** 2).sum().sqrt())
    return cost

# this loss is the sum of negative value of IC over each stock
class IC_loss(nn.Module):
    def __init__(self):
        super(IC_loss, self).__init__()

    def forward(self, logits, target):
        return -pearson_corr(logits, target)


class WeightedICLoss(nn.Module):
    def __init__(self):
        super(WeightedICLoss, self).__init__()

    def forward(self, logits, target):
        return -self.weighted_pearson_corr(logits, target)

    def weighted_pearson_corr(self, x, y):

        w = torch.softmax(x, dim=0)  # Use softmax of logits as weights
        # Compute means
        mean_x = torch.sum(w * x)
        mean_y = torch.sum(w * y)
        # Compute weighted covariance
        cov_xy = torch.sum(w * (x - mean_x) * (y - mean_y))
        # Compute weighted variances
        var_x = torch.sum(w * (x - mean_x) ** 2)
        var_y = torch.sum(w * (y - mean_y) ** 2)
        # Compute weighted Pearson correlation
        corr = cov_xy / torch.sqrt(var_x * var_y)
        
        return corr
    

class SharpeLoss(nn.Module):
    def __init__(self, output_size: int = 1):
        super(SharpeLoss, self).__init__()
        self.output_size = output_size  # in case we have multiple targets => output dim[-1] = output_size * n_quantiles

    def forward(self, weights, y_true):
        captured_returns = weights * y_true
        mean_returns = torch.mean(captured_returns)
        numerator = mean_returns
        denominator = torch.sqrt(torch.mean(captured_returns**2) - mean_returns**2 + 1e-9)
        loss = -(numerator / denominator)
        return loss
    

class ListMLE(nn.Module):
    """
    ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    def __init__(self):
        super(ListMLE, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = y_pred.clone()
        y_true = y_true.clone()
        # shuffle for randomised tie resolution
        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

        mask = y_true_sorted == PADDED_Y_VALUE

        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
        preds_sorted_by_true[mask] = float("-inf")

        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

        observation_loss = torch.log(cumsums + DEFAULT_EPS) - preds_sorted_by_true_minus_max

        observation_loss[mask] = 0.0

        return torch.mean(torch.sum(observation_loss, dim=1))
    

class ListNet(nn.Module):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: loss value, a torch.Tensor
    """
    def __init__(self):
        super(ListNet, self).__init__()

    def forward(self, y_pred, y_true):
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        mask = y_true == PADDED_Y_VALUE
        y_pred[mask] = float('-inf')
        y_true[mask] = float('-inf')

        preds_smax = F.softmax(y_pred, dim=1)
        true_smax = F.softmax(y_true, dim=1)

        preds_smax = preds_smax + DEFAULT_EPS
        preds_log = torch.log(preds_smax)

        return torch.mean(-torch.sum(true_smax * preds_log, dim=1))