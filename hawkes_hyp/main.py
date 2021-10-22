from util import *
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from model import Net
from tqdm import tqdm
import numpy as np
from hgcn.model.base_models import NCModel
import os
import time

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def evaluate():
    model.eval()
    model_hgcn.eval()

    pred_times, pred_events = [], []
    gold_times, gold_events = [], []
    for i, batch in enumerate(tqdm(test_loader)):
        gold_times.append(batch[0][:, -1].numpy())
        gold_events.append(batch[1][:, -1].numpy())
        pred_time, pred_event = model.predict(batch, output, device)
        pred_times.append(pred_time)
        pred_events.append(pred_event)

    pred_events = np.concatenate(pred_events).reshape(-1)
    gold_events = np.concatenate(gold_events).reshape(-1)

    pred_times = np.concatenate(pred_times).reshape(-1)
    gold_times = np.concatenate(gold_times).reshape(-1)
    time_error = rmse_error(pred_times, gold_times)
    mae = abs_error(pred_times,gold_times)
    mape = mean_absolute_percentage_error(pred_times,gold_times)
    acc, recall, f1 = clf_metric(pred_events, gold_events, n_class=config.event_class)
    print(f"epoch {epc}")
    print('mae',mae)
    print("mape",mape)
    print(f"RMSE: {time_error}, PRECISION: {acc}, RECALL: {recall}, F1: {f1}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, default="default")
    parser.add_argument("--data_type", type=str, default='not_graph', help='graph_base')
    parser.add_argument("--dataset", type=str, default='data_so')
    parser.add_argument("--model", type=str, default="Nostradamus")
    parser.add_argument("--model_hgcn", type=str, default="HGCN")
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--emb_dim", type=int, default=10)
    parser.add_argument("--hid_dim", type=int, default=16)
    parser.add_argument("--mlp_dim", type=int, default=16)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.5)  # tradeoff between time and mark loss
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--event_class", type=int, default=22)
    parser.add_argument("--verbose_step", type=int, default=322)
    parser.add_argument("--wd", type=float, default=0.)
    parser.add_argument("--lr", type=int, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--c", type=float, default=1.5)
    parser.add_argument("--SEED", type=int, default=1234)
    parser.add_argument("--manifold", type=str, default="PoincareBall")
    parser.add_argument("--use_bias", type=bool, default="TRUE")
    parser.add_argument("--adam_betas", type=str, default="0.9,0.999")
    parser.add_argument("--act", default='relu')
    parser.add_argument("--task", default='relu')
    parser.add_argument("--bias", default=1)
    parser.add_argument("--use-att", default=0)
    parser.add_argument("--pos_weight", type=int, default=0)
    parser.add_argument("--sgd", action='store_true')
    parser.add_argument(
        "--cell_type", choices=("hyp_gru", "eucl_rnn", "eucl_gru"), default="hyp_gru"
    )
    config = parser.parse_args()

    """The code below is used to set up customized training device on computer"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("You are using GPU acceleration.")
        print("Number of CUDAs(cores): ", torch.cuda.device_count())
    else:
        device = torch.device("cpu")
        print("CUDA is not Available. You are using CPU only.")
        print("Number of cores: ", os.cpu_count())

    weight = np.ones(config.event_class)


    train_set = Dataset(config, subset='train')
    test_set = Dataset(config, subset='test')
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=False, collate_fn=Dataset.to_features)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=Dataset.to_features)

    model = Net(config, device)
    model.set_optimizer()

    id_process = os.getpid()

    log_file_name = "./log/" + config.dataset + str(config.SEED) + str(config.batch_size) + ".txt"
    log = open(log_file_name, 'w')

    model_hgcn = NCModel(config)
    time_begin = time.time()
    for epc in range(config.epochs):
        model_hgcn.train()
        model.train()
        range_loss1 = range_loss2 = range_loss = 0

        if config.data_type == 'graph_base':
            adj = read_adj(config.dataset)
            fea = fea_mat(config.dataset, 'train')

        else:
            adj = get_adj(train_set.generate_adj())
            fea = fea_mat(config.dataset, 'train')
            print("ad",adj.shape)
            print('f',fea.shape)

        for i, batch in enumerate(tqdm(train_loader)):
            if type(fea) is np.ndarray:
                fea = torch.from_numpy(fea).float()
            hgcn_out = model_hgcn.encode(x=fea, adj=adj)
            output = model_hgcn.decode(hgcn_out, adj)
            l1, l2, l, = model.train_batch(batch, output, device)
            range_loss1 += l1
            range_loss2 += l2
            range_loss += l

            if (i + 1) % config.verbose_step == 0:
                print("time loss: ", range_loss1 / config.verbose_step)
                print("event loss:", range_loss2 / config.verbose_step)
                print("total loss:", range_loss / config.verbose_step)
                log.write(str(epc) + ',')
                log.write(str(range_loss / config.verbose_step) + ',')
                log.write(str(range_loss1 / config.verbose_step) + ',')
                log.write(str(range_loss2 / config.verbose_step))
                log.write("\n")
                range_loss1 = range_loss2 = range_loss = 0
        evaluate()
    time_end = time.time()
    print("time",time_end-time_begin)
    log.close()
