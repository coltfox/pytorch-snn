from spiking_layers.dense_spiking_layer import DenseSpikingLayer
from spiking_layers.rate_encoder import RateEncoder
from dense_snn import DenseSNN
from train_eval_snn import TrainEvalDenseSNN

DEVICE = 'cpu'
BATCH_SIZE = 500
N_TS = 25
EPOCHS = 10

model = DenseSNN(
    n_ts=N_TS,
    input_encoder=RateEncoder(
        n_neurons=784, batch_size=BATCH_SIZE, device=DEVICE),
    hidden_layers=[
        DenseSpikingLayer(n_in=784, n_out=1024,
                          batch_size=BATCH_SIZE, device=DEVICE),
        DenseSpikingLayer(n_in=1024, n_out=512,
                          batch_size=BATCH_SIZE, device=DEVICE)
    ],
    output_layer=DenseSpikingLayer(
        n_in=512, n_out=10, batch_size=BATCH_SIZE, device=DEVICE)
)

train_eval_model = TrainEvalDenseSNN(
    model=model,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    device=DEVICE
)

train_eval_model.train()
train_eval_model.eval()
