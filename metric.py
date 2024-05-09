from torchmetrics import Metric
import torch
from sklearn.metrics import f1_score

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__(dist_sync_on_step=False)
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        preds = preds.argmax(dim=1)
        if preds.shape != target.shape:
            raise ValueError("Predictions and targets must have the same shape")
        self.correct += (preds == target).sum()
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()

class MyF1Score(Metric):
    def __init__(self, num_classes):
        super().__init__(dist_sync_on_step=False)
        self.num_classes = num_classes
        self.add_state("all_targets", default=[], dist_reduce_fx="cat")
        self.add_state("all_preds", default=[], dist_reduce_fx="cat")

    def update(self, preds, target):
        preds = torch.argmax(preds, dim=1)
        self.all_preds.append(preds)
        self.all_targets.append(target)

    def compute(self):
        y_pred = torch.cat(self.all_preds)
        y_true = torch.cat(self.all_targets)
        f1_scores = {}
        
        for cls in range(self.num_classes):
            y_true_cls = (y_true == cls).int()
            y_pred_cls = (y_pred == cls).int()
            f1 = f1_score(y_true_cls.cpu().numpy(), y_pred_cls.cpu().numpy(), zero_division=1)
            f1_scores[cls] = f1
        
        return f1_scores

# 사용 예시
num_classes = 3
accuracy_metric = MyAccuracy()
f1_score_metric = MyF1Score(num_classes=num_classes)
