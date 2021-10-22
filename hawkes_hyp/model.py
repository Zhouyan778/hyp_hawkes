from torch import nn
from util import *
import geoopt.manifolds.poincare.math as pmath
import hyrnn
import geoopt
from hgcn.layers import hyp_layers


def RMSE_error(pred, gold):
    return np.sqrt(np.mean((pred - gold) ** 2))


class Net(nn.Module):
    def __init__(self, config, device):
        super(Net, self).__init__()
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)

        self.config = config
        self.n_class = config.event_class
        self.c = config.c
        self.manifold = config.manifold
        self.use_bias = config.use_bias
        self.dropout = config.dropout
        self.sgd = config.sgd

        self.embed = nn.Linear(in_features=self.n_class, out_features=1)
        self.emb_drop = nn.Dropout(p=config.dropout)

        self.projector_source = hyrnn.MobiusLinear(
            1, 1, c=1.0
        )
        self.hyp_linear = hyp_layers.HypLinear(self.manifold,
                                               in_features=self.n_class,
                                               out_features=1,
                                               c=self.c,
                                               dropout=self.dropout,
                                               use_bias=self.use_bias)
        self.hyp_gru = hyrnn.nets.MobiusGRU(input_size=24,
                                            hidden_size=config.hid_dim,
                                            num_layers=1,
                                            c=1.0,
                                            hyperbolic_input=True
                                            )

        self.gru1 = nn.GRUCell(input_size=self.n_class + self.config.emb_dim + 1,
                               hidden_size=config.hid_dim)
        self.Wy = nn.Parameter(torch.ones(config.emb_dim, config.hid_dim) * 0.0)
        self.Wh = torch.eye(config.hid_dim, requires_grad=True)
        self.Wt = torch.ones((1, config.hid_dim), requires_grad=True)  # * 1e-3
        self.Vy = nn.Parameter(torch.ones(config.hid_dim, self.n_class) * 1e-3)
        self.Vt = nn.Parameter(torch.ones(config.hid_dim, 1) * 1e-3)
        self.wt = nn.Parameter(torch.tensor(1.0))
        self.bh = nn.Parameter(torch.log(torch.ones(1, 1)))
        self.bk = nn.Parameter(torch.ones((1, self.n_class)) * 0.0)
        self.gru = nn.GRU(input_size=24,
                          hidden_size=config.hid_dim,
                          batch_first=True)
        self.mlp = nn.Linear(in_features=config.hid_dim, out_features=config.mlp_dim)
        self.mlp_drop = nn.Dropout(p=config.dropout)

        self.hyn_embedding = hyrnn.LookupEmbedding(
            config.event_class,
            1,
            manifold=geoopt.PoincareBall(c=self.c),
        )
        self.embedding = nn.Embedding(num_embeddings=config.event_class, embedding_dim=1)
        with torch.no_grad():
            self.hyn_embedding.weight.set_(
                pmath.expmap0(self.hyn_embedding.weight.normal_() / 10, c=self.c)
            )
        self.event_linear = nn.Linear(in_features=config.mlp_dim, out_features=config.event_class)
        self.time_linear = nn.Linear(in_features=config.mlp_dim, out_features=1)

        self.set_criterion()

    def set_optimizer(self):
        adam_betas = self.config.adam_betas.split(",")
        if not self.sgd:
            self.optimizer = geoopt.optim.RiemannianAdam(
                self.parameters(),
                lr=self.config.lr,
                betas=(float(adam_betas[0]), float(adam_betas[1])),
                stabilize=10,
                weight_decay=self.config.wd
            )
        else:
            self.optimizer = geoopt.optim.RiemannianSGD(
                self.parameters(),
                lr=self.config.lr,
                stabilize=10,
                weight_decay=self.config.wd)

    def set_criterion(self):
        self.event_criterion = nn.CrossEntropyLoss()
        if self.config.model == 'Nostradamus':
            self.intensity_w = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
            self.intensity_b = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
            self.time_criterion = self.NostradamusLoss
        else:
            self.time_criterion = nn.MSELoss()

    def NostradamusLoss(self, hidden_j, time_duration):
        loss = torch.mean(hidden_j + self.intensity_w * time_duration + self.intensity_b +
                          (torch.exp(hidden_j + self.intensity_b) -
                           torch.exp(
                               hidden_j + self.intensity_w * time_duration + self.intensity_b)) / self.intensity_w)
        return -loss

    def forward(self, input_time, event_input, gcn_out):
        event_input1 = event_input.reshape(-1)
        node_information = self.embedding(event_input1)
        node_information = node_information.reshape(self.config.batch_size, -1, 1)
        # gcn input
        input_event = event_input.reshape(-1)
        out = torch.index_select(gcn_out, 0, input_event)
        out = pmath.expmap0(out)
        out = out.reshape(input_time.shape[0], input_time.shape[1], -1)
        out = out / 10
        input_time = pmath.expmap0(input_time) / 10

        lstm_input = torch.cat((out, node_information), dim=-1)
        lstm_input = torch.cat((lstm_input, input_time.unsqueeze(-1)), dim=-1)
        lstm_input = lstm_input.permute(1, 0, 2)
        hidden_state, _ = self.hyp_gru(lstm_input)
        mlp_output = torch.tanh(self.mlp(hidden_state[-1, :, :]))
        mlp_output = self.mlp_drop(mlp_output)
        event_out = self.event_linear(mlp_output)
        time_out = self.time_linear(mlp_output)
        return event_out, time_out

    def dispatch(self, tensors):
        for i in range(len(tensors)):
            tensors[i] = tensors[i].contiguous()
        return tensors

    def train_batch(self, batch, out, device):
        time_tensor, event_tensor = batch
        time_tensor.to(device)
        event_tensor.to(device)

        # TODO 写入的duration 有误
        time_input, time_duration = self.dispatch([time_tensor[:, :-1], time_tensor[:, -1]])
        event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])
        event_out, time_out = self.forward(time_input, event_input, out)

        # TODO 输入应该为时间差
        loss1 = self.time_criterion(time_out.view(-1), time_duration.view(-1))
        loss2 = self.event_criterion(event_out.view(-1, self.n_class), event_target.view(-1))
        loss = self.config.alpha * loss1 + loss2
        loss.backward(retain_graph=True)

        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss1.item(), loss2.item(), loss.item()

    def predict(self, batch, out, device):
        time_tensor, event_tensor = batch
        time_tensor.to(device)
        event_tensor.to(device)
        time_input, time_duration = self.dispatch([time_tensor[:, :-1], time_tensor[:, 1:]])
        event_input, event_target = self.dispatch([event_tensor[:, :-1], event_tensor[:, -1]])

        event_out, time_out = self.forward(time_input, event_input, out)
        x = torch.log(torch.exp(time_out + self.intensity_b) + self.intensity_w * np.log(2)) - (
                time_out + self.intensity_b)
        x = torch.squeeze(x)
        duration = x / torch.squeeze(self.intensity_w)
        time_pred = duration

        event_pred = np.argmax(event_out.detach().numpy(), axis=-1)
        time_pred = np.array(time_pred.detach().numpy())

        return time_pred, event_pred
