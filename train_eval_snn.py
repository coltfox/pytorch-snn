import torch
import time
import torchvision
import numpy as np
from dense_snn import DenseSNN


class TrainEvalDenseSNN(object):
    def __init__(self, model: DenseSNN, epochs=10, batch_size: int = 500, device: torch.device | str = 'cpu'):
        """
        Args:
          n_ts <int>: Number of time-steps to present each image for training/test.
          epochs <int>: Number of training epochs.
        """
        self.epochs = epochs
        self.device = device
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.model = model.to(device)  # n_ts = presentation time-steps.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        # Get the Train- and Test- Loader of the MNIST dataset.
        self.train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           # ToTensor transform automatically converts all image pixels in range [0, 1].
                                           torchvision.transforms.ToTensor()
                                       ])
                                       ),
            batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                       transform=torchvision.transforms.Compose([
                                           # ToTensor transform automatically converts all image pixels in range [0, 1].
                                           torchvision.transforms.ToTensor()
                                       ])
                                       ),
            batch_size=batch_size, shuffle=True)

    def train(self, epoch):
        all_true_ys, all_pred_ys = [], []
        all_batches_loss = []
        self.model.train()
        for trn_x, trn_y in self.train_loader:
            # Each batch trn_x and trn_y is of shape (batch_size, 1, 28, 28) and
            # (batch_size) respectively, where the image pixel values are between
            # [0, 1], and the image class is a numeral between [0, 9].
            # Flatten from dim 1 onwards.
            trn_x = trn_x.flatten(start_dim=1).to(self.device)
            # Output = (batch_size, n_ts, #Classes).
            all_ts_out_spks = self.model(trn_x)
            # Mean over time-steps.
            mean_spk_rate_over_ts = torch.mean(all_ts_out_spks, axis=1)
            # Shape of mean_spk_rate_all_ts is (batch_size, #Classes).
            # ArgMax over classes.
            trn_preds = torch.argmax(mean_spk_rate_over_ts, axis=1)
            # Shape of trn_preds is (batch_size,).
            all_true_ys.extend(trn_y.detach().numpy().tolist())
            all_pred_ys.extend(trn_preds.detach().numpy().tolist())

            # Compute Training Loss and Back-propagate.
            loss_value = self.loss_function(mean_spk_rate_over_ts, trn_y)
            all_batches_loss.append(loss_value.detach().item())
            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

        trn_accuracy = np.mean(np.array(all_true_ys) == np.array(all_pred_ys))
        return trn_accuracy, np.mean(all_batches_loss)

    def eval(self, epoch):
        all_true_ys, all_pred_ys = [], []
        self.model.eval()
        with torch.no_grad():
            for tst_x, tst_y in self.test_loader:
                # Each batch tst_x and tst_y is of shape (batch_size, 1, 28, 28) and
                # (batch_size) respectively, where the image pixel values are between
                # [0, 1], and the image class is a numeral between [0, 9].
                # Flatten from dim 1 onwards.
                tst_x = tst_x.flatten(start_dim=1).to(self.device)
                all_ts_out_spks = self.model(tst_x)
                mean_spk_rate_over_ts = torch.mean(all_ts_out_spks, axis=1)
                tst_preds = torch.argmax(mean_spk_rate_over_ts, axis=1)
                all_true_ys.append(tst_y.detach().numpy().tolist())
                all_pred_ys.append(tst_preds.detach().numpy().tolist())

        tst_accuracy = np.mean(np.array(all_true_ys) == np.array(all_pred_ys))
        return tst_accuracy

    def train_eval(self):
        start = time.time()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()

            _, mean_loss = self.train(epoch)
            tst_accuracy = self.eval(epoch)

            epoch_end = time.time()

            print("Epoch: %s, Training Loss: %s, Test Accuracy: %s, Time Elapsed: %ss"
                  % (epoch, mean_loss, tst_accuracy, round(epoch_end - epoch_start, 3)))

        end = time.time()

        print("Finished training in %ss" % (round(end - start, 3)))
