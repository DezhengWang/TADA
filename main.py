import argparse
from utils.tools import *
from torch import optim
from utils.data_loader import *
from tqdm import tqdm
import time
from Models import TADA, INFO, LSTNet, TCN, Transformer
from thop import profile

parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--datasets', type=str, default="SML2010", help='name of datasets')
parser.add_argument('--net', type=str, default="our", help='name of net')
parser.add_argument('--seq_len', type=int, default=96, help='# sequence length')
parser.add_argument('--label_len', type=int, default=48, help='# sequence label')
parser.add_argument('--pred_len', type=int, default=24, help='# sequence prediction')
parser.add_argument('--feature', type=int, default=22, help='# feature')
parser.add_argument('--hid', type=int, default=512, help='number of RNN hidden units')
parser.add_argument('--clip', type=float, default=10., help='gradient clipping')
parser.add_argument('--epochs', type=int, default=10, help='upper epoch limit')
parser.add_argument('--MultiStep', type=bool, default=False, help='Multi-step prediction')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--pred_horizon', type=int, default=1, help='scope of prediction horizon')
parser.add_argument('--n_heads', type=int, default=8, help='# attention head')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--freq', type=str, default="h")
args = parser.parse_args([])

for args.seq_len in [96]:
    for args.datasets in ["ETTh"]:
        for args.net in ["TCN", "LSTNet", "INFO", "Transformer", "OUR"]:  #

            tag = args.net + "Ms" if args.MultiStep else args.net
            args.trans = None
            tag = tag + "s" if args.trans else tag

            if args.datasets == "ETTh":
                Dataset = Dataset_ETT_hour
                args.feature = 7
            elif args.datasets == "ETTm":
                Dataset = Dataset_ETT_min
                args.feature = 7
            elif args.datasets == "ELD":
                Dataset = Dataset_ELD_hour
                args.feature = 315
            elif args.datasets == "Steel":
                Dataset = Dataset_STI_hour
                args.feature = 9
            elif args.datasets == "PCT":
                Dataset = Dataset_PCT_hour
                args.feature = 8

            if args.MultiStep:
                args.pred_horizon = args.pred_len

            e_layers = 3
            d_layers = 2
            for e_layers in range(3, 4):
                for d_layers in range(2, 3):
                    print(f"[INFO] " + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) +
                          f" The {tag} is training on the Device: {try_gpu()}, Task {args.datasets}")
                    print(f"[INFO] " + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) +
                          f" Encoders: {e_layers}  Dncoders: {d_layers}")

                    for index in range(20):  #
                        args.lr = 1e-4
                        args.pred_len = 24
                        print(f"[INFO] " + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) +
                              f" # data in encoder: {args.seq_len}" +
                              f" # data in decoder: {args.label_len}" +
                              f" # data for prediction: {args.pred_len}")


                        def process(net, data, pred_len, label_len, output_attention=False):
                            batch_x, batch_y, batch_x_mark, batch_y_mark = data
                            dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[-1]]).float().to(try_gpu())
                            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float()

                            batch_y = batch_y[:, label_len, :].unsqueeze(1)
                            # encoder - decoder
                            if output_attention:
                                outputs, attns = net(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                                return outputs, batch_y, attns
                            outputs = net(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            return outputs, batch_y


                        def inference_step(net, loader, process, args, output_attention=False):
                            net.eval()
                            net.to(try_gpu())
                            preds = []
                            trues = []
                            for k, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                                batch_x, batch_x_mark = batch_x.to(try_gpu()), batch_x_mark.to(try_gpu())
                                batch_y, batch_y_mark = batch_y.to(try_gpu()), batch_y_mark.to(try_gpu())
                                if k % args.pred_len == 0:
                                    for ii in range(args.pred_len):
                                        res = process(net, (batch_x[:, -args.seq_len:],
                                                            batch_y[:, ii:ii + args.label_len + args.pred_horizon],
                                                            batch_x_mark[:, -args.seq_len:],
                                                            batch_y_mark[:,
                                                            ii:ii + args.label_len + args.pred_horizon]),
                                                      pred_len=args.pred_horizon,
                                                      label_len=args.label_len,
                                                      output_attention=output_attention)
                                        preds.append(res[0].detach().cpu().numpy())
                                        trues.append(res[1].detach().cpu().numpy())
                                        batch_x = torch.cat((batch_x, res[0]), dim=1)
                                        batch_y = torch.cat(
                                            (batch_x, batch_y[:, ii + args.label_len + args.pred_horizon:]), dim=1)
                                        batch_x_mark = torch.cat((batch_x_mark,
                                                                  batch_y_mark[:, ii + args.label_len].unsqueeze(1)),
                                                                 dim=1)

                            preds = np.array(preds)
                            trues = np.array(trues)
                            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
                            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
                            if output_attention:
                                return preds, trues, res[2]
                            return preds, trues


                        def inference_multistep(net, loader, process, args, output_attention=False):
                            net.eval()
                            net.to(try_gpu())
                            preds = []
                            trues = []
                            for k, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                                batch_x, batch_x_mark = batch_x.to(try_gpu()), batch_x_mark.to(try_gpu())
                                batch_y, batch_y_mark = batch_y.to(try_gpu()), batch_y_mark.to(try_gpu())
                                if k % args.pred_len == 0:
                                    res = process(net, (batch_x,
                                                        batch_y,
                                                        batch_x_mark,
                                                        batch_y_mark),
                                                  pred_len=args.pred_len,
                                                  label_len=args.label_len,
                                                  output_attention=output_attention)
                                    preds.append(res[0].detach().cpu().numpy())
                                    trues.append(res[1].detach().cpu().numpy())

                            preds = np.array(preds)
                            trues = np.array(trues)
                            preds = preds.reshape(-1, 1, preds.shape[-1])
                            trues = trues.reshape(-1, 1, trues.shape[-1])
                            if output_attention:
                                return preds, trues, res[2]
                            return preds, trues


                        def evaluate(loader, net, criterion, args):
                            if args.MultiStep:
                                preds, trues = inference_multistep(net, loader, process, args)
                            else:
                                preds, trues = inference_step(net, loader, process, args)
                            loss = criterion(torch.from_numpy(preds), torch.from_numpy(trues))
                            return np.average(loss)


                        def train(train_loader, net, criterion, opt, args):
                            net.train()
                            net.to(try_gpu())
                            train_loss = []
                            for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
                                batch_x, batch_x_mark = batch_x.to(try_gpu()), batch_x_mark.to(try_gpu())
                                batch_y, batch_y_mark = batch_y.to(try_gpu()), batch_y_mark.to(try_gpu())
                                opt.zero_grad()
                                pred, true = process(net, (batch_x, batch_y[:, :args.label_len + args.pred_horizon],
                                                           batch_x_mark,
                                                           batch_y_mark[:, :args.label_len + args.pred_horizon]),
                                                     pred_len=args.pred_horizon, label_len=args.label_len)
                                pred = pred.type(true.dtype)
                                loss = criterion(pred, true)
                                train_loss.append(loss.item())
                                loss.backward()
                                opt.step()
                            return np.average(train_loss)


                        train_loader, eval_loader, test_loader = load_dataloader(
                            args, Dataset, root_path="./datasets/" + args.datasets + "/", transform=args.trans)

                        root_path = mkdir(pred_len="Pred_" + str(args.seq_len) + "_" + str(args.label_len),
                                          model=tag + "_e" + str(e_layers) + "d" + str(d_layers),
                                          augmentation=args.trans.__name__,
                                          dataset=args.datasets, index=index)

                        net = eval(args.net).Model(enc_in=args.feature, dec_in=args.feature, c_out=args.feature,
                                                   out_len=args.pred_horizon, d_model=args.hid, n_heads=args.n_heads,
                                                   e_layers=e_layers, d_layers=d_layers, d_ff=args.hid,
                                                   dropout=0.1, embed='fixed', freq=args.freq, activation='relu',
                                                   output_attention=True, mix=True, args=args)

                        criterion = nn.MSELoss()
                        opt = optim.Adam(net.parameters(), lr=args.lr)

                        nParams = sum([p.nelement() for p in net.parameters()])
                        print(f'[INFO] ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) +
                              ' * number of parameters: %d' % nParams, ", Learning rate: {:.1e}".format(args.lr))

                        loop = tqdm(range(1, args.epochs + 1))
                        losses = []
                        times = []
                        best_eval = 1000
                        for epoch in loop:
                            start_time = time.time()
                            train_loss = train(train_loader, net, criterion, opt, args)
                            during_time = time.time() - start_time
                            eval_losses = evaluate(eval_loader, net, criterion, args)
                            adjust_learning_rate(opt, epoch + 1, args)
                            if eval_losses < best_eval:
                                save_model(net=net, path=root_path)
                                best_eval = eval_losses
                            # log loss
                            losses.append([train_loss, eval_losses])
                            times.append(round(during_time, 2))

                            loop.set_description(
                                f'[INFO] {time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())} {tag},[{index}]')
                            loop.set_postfix(
                                loss='Train: {:.4f}, Eval: {:.4f}, lr {:.1e})'.format(train_loss, eval_losses, args.lr))

                        # load best model
                        with open(os.path.join(root_path, "net.pt"), 'rb') as f:
                            net = torch.load(f)

                        print(f'[INFO] ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) +
                              f' Training {tag} on {args.datasets} with Pred({args.pred_len}),' +
                              'Best Eval Loss[{:.4f}]'.format(best_eval))

                        test_criterion = nn.MSELoss()
                        for pred_len in [24, 48, 96, 192, 336, 480]:  # , 624, 720
                            args.pred_len = pred_len
                            test_loader = load_dataloader(args, Dataset, root_path="./datasets/" + args.datasets + "/")[
                                -1]
                            # save test result
                            if args.MultiStep:
                                Y_hat, Y, attns = inference_multistep(net, test_loader, process, args,
                                                                      output_attention=True)
                            else:
                                Y_hat, Y, attns = inference_step(net, test_loader, process, args, output_attention=True)
                            test_eval = test_criterion(torch.from_numpy(Y_hat), torch.from_numpy(Y)).item()
                            print(f'[INFO] ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) +
                                  f' Testing {tag} on {args.datasets} with Pred({args.pred_len}),' +
                                  'Test Loss[{:.4f}]'.format(test_eval))
                            np.savez(os.path.join(root_path, "result_" + str(args.pred_len) + ".npz"),
                                     y=Y, y_hat=Y_hat,
                                     loss=np.array(losses),
                                     attns=[attn.detach().cpu() if attn is not None else None for attn in attns],
                                     nParams=nParams,
                                     times=round(np.array(times).mean(), 2), )
                        print("\n")
print(f"[INFO] " + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + " Training is Done!")
