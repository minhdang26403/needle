import sys

sys.path.append("../python")
import needle.data as data
import needle.nn as nn
import needle.optim as optim
import numpy as np

np.random.seed(0)


def ResidualBlock(
    dim: int,
    hidden_dim: int,
    norm: type[nn.BatchNorm1d] | type[nn.LayerNorm1d],
    drop_prob: float = 0.1,
) -> nn.Module:
    """Residual block with optional batch normalization and dropout."""

    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim),
            )
        ),
        nn.ReLU(),
    )


def MLPResNet(
    dim: int,
    hidden_dim: int = 100,
    num_blocks: int = 3,
    num_classes: int = 10,
    norm: type[nn.BatchNorm1d] | type[nn.LayerNorm1d] = nn.BatchNorm1d,
    drop_prob: float = 0.1,
) -> nn.Module:
    """MLP-ResNet model with optional batch normalization and dropout."""

    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *(
            ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob)
            for _ in range(num_blocks)
        ),
        nn.Linear(hidden_dim, num_classes),
    )


def epoch(
    dataloader: data.DataLoader,
    model: nn.Module,
    opt: optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Train the model for one epoch."""

    np.random.seed(4)

    model.train() if opt else model.eval()
    error_count = 0
    num_samples = 0

    loss_sum = 0
    num_batches = 0
    f = nn.SoftmaxLoss()

    for batch_x, batch_y in dataloader:
        out = model(batch_x)
        error_count += (np.argmax(out.numpy(), axis=1) != batch_y.numpy()).sum()
        num_samples += batch_x.shape[0]
        loss = f(out, batch_y)
        num_batches += 1
        if opt:
            loss.backward()
            opt.step()
        loss_sum += loss.numpy().item()

    return error_count / num_samples, loss_sum / num_batches


def train_mnist(
    batch_size: int = 100,
    epochs: int = 10,
    optimizer: type[optim.Adam] | type[optim.SGD] = optim.Adam,
    lr: float = 0.001,
    weight_decay: float = 0.001,
    hidden_dim: int = 100,
    data_dir: str = "data",
) -> tuple[float, float, float, float]:
    """Train the model for multiple epochs."""

    np.random.seed(4)
    train_dataset = data.MNISTDataset(
        f"{data_dir}/train-images-idx3-ubyte.gz", "./data/train-labels-idx1-ubyte.gz"
    )
    train_dataloader = data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_dataset = data.MNISTDataset(
        f"{data_dir}/t10k-images-idx3-ubyte.gz", "./data/t10k-labels-idx1-ubyte.gz"
    )
    test_dataloader = data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    training_error = 0.0
    training_loss = 0.0
    for _ in range(epochs):
        training_error, training_loss = epoch(train_dataloader, model, opt)

    model.eval()
    test_error, test_loss = epoch(test_dataloader, model)
    return training_error, training_loss, test_error, test_loss


if __name__ == "__main__":
    train_mnist(data_dir="../data")
