import torch
from torch import nn
from pytorch_lightning import LightningModule, Trainer
from torchvision.models import resnet18, ResNet18_Weights
from torchmetrics import Metric
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset, random_split

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__(dist_sync_on_step=False)
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        preds = torch.argmax(preds, dim=1)
        self.correct += torch.sum(preds == targets)
        self.total += targets.numel()

    def compute(self):
        return self.correct.float() / self.total

class MyF1Score(Metric):
    def __init__(self, num_classes):
        super().__init__(dist_sync_on_step=False)
        self.num_classes = num_classes
        self.add_state("preds", default=torch.tensor([]), dist_reduce_fx="cat")
        self.add_state("targets", default=torch.tensor([]), dist_reduce_fx="cat")

    def update(self, preds, targets):
        preds = torch.argmax(preds, dim=1)
        self.preds = torch.cat([self.preds, preds])
        self.targets = torch.cat([self.targets, targets])

    def compute(self):
        if self.preds.nelement() == 0 or self.targets.nelement() == 0:
            return torch.tensor(0.0)
        try:
            _, _, f1, _ = precision_recall_fscore_support(self.targets.cpu(), self.preds.cpu(), average='macro')
        except ValueError:
            f1 = 0.0  # Fallback to zero if insufficient data for F1 calculation
        return torch.tensor(f1)

class SimpleClassifier(LightningModule):
    def __init__(self, model_name='resnet18', num_classes=4, optimizer_params=None, scheduler_params=None):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = MyAccuracy()
        self.f1_score = MyF1Score(num_classes)
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss_fn(preds, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), **(self.optimizer_params or {'lr': 0.001}))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **(self.scheduler_params or {'step_size': 10, 'gamma': 0.1}))
        return [optimizer], [scheduler]

# Data loader creation
def create_data_loaders():
    dataset = TensorDataset(torch.randn(1000, 3, 224, 224), torch.randint(0, 4, (1000,)))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    return train_loader, val_loader

train_loader, val_loader = create_data_loaders()

# Create an instance of SimpleClassifier
model = SimpleClassifier(model_name='resnet18', num_classes=4)

# Trainer setup and execution
trainer = Trainer(
    max_epochs=500,
    devices=1,  # Use 1 GPU. To use all GPUs, set devices=-1
    accelerator='gpu'
)

try:
    trainer.fit(model, train_loader, val_loader)
    trainer.validate(model, val_loader)
except Exception as e:
    print(f"An error occurred: {e}")
