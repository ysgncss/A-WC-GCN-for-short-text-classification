import argparse
import numpy as np
import torch
import torch as th
import torch.nn.functional as F
import torch.nn.init as INIT
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader
import dgl

from MyDataset import MyDataset
from RGCN import HeteroClassifier


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = th.tensor(labels)
    return batched_graph, batched_labels


def main(args):
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)

    # GPU setting
    cuda = args.gpu >= 0
    device = th.device('cuda:{}'.format(args.gpu)) if cuda else th.device('cpu')
    if cuda:
        th.cuda.set_device(args.gpu)

    # dataset and class_unm
    dataset = args.dataset
    if dataset == "TREC":
        class_num = 6
    elif dataset == "AGNews":
        class_num = 4
    elif dataset == "MR":
        class_num = 2
    elif dataset == "Ohsumed":
        class_num = 23
    elif dataset == "R8":
        class_num = 8
    elif dataset == "R52":
        class_num = 52
    elif dataset == "SST1":
        class_num = 5
    elif dataset == "SST2":
        class_num = 2
    elif dataset == "WebKB":
        class_num = 8
    elif dataset == "wiki":
        class_num = 10
    elif dataset == "Dblp":
        class_num = 6
    elif dataset == "20ng":
        class_num = 20
    elif dataset == "aclimdb":
        class_num = 2
    elif dataset == "Snippets":
        class_num = 8
    elif dataset == "aclimdb":
        class_num = 41

    elif dataset == "AmazonReview":
        class_num = 5
    elif dataset == "YahooAnswers":
        class_num = 10
    elif dataset == "Dbpedia":
        class_num = 14

    elif dataset == "Stackoverflow":
        class_num = 20
    elif dataset == "Biomedical":
        class_num = 20
    elif dataset == "SougouNews":
        class_num = 5
    elif dataset == "Twitter":
        class_num = 2
    elif dataset == "TagMyNews":
        class_num = 7

    testset = MyDataset(sub='test', name=dataset, concept=args.concept, neighbor=args.neighbor, class_num=class_num)
    test_loader = DataLoader(
        dataset=testset,
        batch_size=args.batch_size,
        collate_fn=collate,
        shuffle=False,
        num_workers=0
    )

    devset = MyDataset(sub='dev', name=dataset, concept=args.concept, neighbor=args.neighbor, class_num=class_num)
    dev_loader = DataLoader(
        dataset=devset,
        batch_size=args.batch_size,
        collate_fn=collate,
        shuffle=False,
        num_workers=0
    )

    trainset = MyDataset(sub='train', name=dataset, concept=args.concept, neighbor=args.neighbor, class_num=class_num)
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        collate_fn=collate,
        shuffle=True,
        num_workers=0
    )

    if args.model == "WC-GCN":
        model = HeteroClassifier(
            in_dim=args.x_size,
            hidden_dim=args.h_size,
            n_classes=class_num,
            rel_names=['A', 'B', 'C'],
            dataset=dataset,
            layer= args.layer
        ).to(device)


    params_ex_emb = [x for x in list(model.parameters()) if
                     x.requires_grad and x.size(-1) != 1 and x.size(-1) != args.x_size]
    for p in params_ex_emb:
        if p.dim() > 1:
            INIT.xavier_uniform_(p)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    result = open("result.txt", "a", encoding="utf-8")
    all_best_dev_acc = 0

    total_epoch = 0  # 记录进行到多少batch
    last_bset_epoch = 0

    model_path = trainset.model_path

    for epoch in range(args.epochs):
        total_epoch += 1
        # early stop
        if last_bset_epoch >= args.early_stop_epoches:
            break
        # train
        model.train()
        train_predict_all = np.array([], dtype=int)
        train_labels_all = np.array([], dtype=int)
        train_loss_total = 0
        for step, batch in enumerate(train_loader):
            batched_graph = batch[0]
            labels = batch[1]
            labels = labels.to(device)
            g = batched_graph.to(device)
            # print(g)
            logits = model(g)
            loss = F.cross_entropy(logits, labels)
            train_loss_total += loss
            pred = th.argmax(logits, 1).data.cpu()
            true = labels.data.cpu()
            train_labels_all = np.append(train_labels_all, true)
            train_predict_all = np.append(train_predict_all, pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = metrics.accuracy_score(train_labels_all, train_predict_all)

        # dev
        model.eval()
        dev_predict_all = np.array([], dtype=int)
        dev_labels_all = np.array([], dtype=int)
        dev_loss_total = 0
        for step, batch in enumerate(dev_loader):
            batched_graph = batch[0]
            labels = batch[1]
            labels = labels.to(device)
            g = batched_graph.to(device)
            with th.no_grad():
                logits = model(g)
                loss = F.cross_entropy(logits, labels)
                dev_loss_total += loss
                dev_pred = th.argmax(logits, 1).data.cpu()
                dev_true = labels.data.cpu()
                dev_labels_all = np.append(dev_labels_all, dev_true)
                dev_predict_all = np.append(dev_predict_all, dev_pred)
        dev_acc = metrics.accuracy_score(dev_labels_all, dev_predict_all)

        # save model by acc
        if dev_acc < all_best_dev_acc:
            last_bset_epoch += 1
            improve = ''
        else:
            improve = '*'
            all_best_dev_acc = dev_acc
            last_bset_epoch = 0
            # save model
            torch.save(model.state_dict(), model_path)

        msg = 'epoch: {0},  Train Loss: {1},  Train Acc: {2},  Dev Loss: {3},  Dev Acc: {4},  {5}'
        print(msg.format(epoch, train_loss_total, train_acc, dev_loss_total, dev_acc, improve))

        scheduler.step()

    # test
    model.load_state_dict(torch.load(model_path))
    model.eval()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    test_loss_total = 0
    for step, batch in enumerate(test_loader):
        batched_graph = batch[0]
        labels = batch[1]
        labels = labels.to(device)
        g = batched_graph.to(device)
        with th.no_grad():
            logits = model(g)
            loss = F.cross_entropy(logits, labels)
            test_loss_total += loss
            pred = th.argmax(logits, 1).data.cpu()
            true = labels.data.cpu()
            labels_all = np.append(labels_all, true)
            predict_all = np.append(predict_all, pred)

    test_acc = metrics.accuracy_score(labels_all, predict_all)

    msg = 'lr: {0}, batch-size: {1}, h-size: {2}, weight-decay: {3}, concept: {4}, test_acc:{5}, dataset:{6}, ' \
          'neighbor:{7} '
    print(msg.format(args.lr, args.batch_size, args.h_size, args.weight_decay, args.concept, test_acc, args.dataset,
                     args.neighbor))
    result.write(
        msg.format(args.lr, args.batch_size, args.h_size, args.weight_decay, args.concept, test_acc, args.dataset,
                   args.neighbor) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--model', type=str, default='WC-GCN')
    parser.add_argument('--layer', type=int, default=1)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--x-size', type=int, default=300)
    parser.add_argument('--h-size', type=int, default=300)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--early-stop-epoches', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--concept', type=int, default=4)
    parser.add_argument('--dataset', type=str, default="MR")
    parser.add_argument('--neighbor', type=int, default=1)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    args = parser.parse_args()
    print(args)
    main(args)
