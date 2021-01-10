import json
import os
from collections import OrderedDict
import dgl
import nltk
import numpy as np
import torch.nn as nn
import torch
from dgl.data import DGLDataset
import networkx as nx
from dgl import save_graphs, load_graphs
import pickle as pkl

UNK = '<UNK>'  # 未知字


class MyDataset(DGLDataset):

    def __init__(self, sub='train', name='SST2', concept=1, neighbor=1, class_num=23):
        self.sub = sub
        self.neighbor = neighbor
        self.concept = concept
        self.class_num = class_num
        self.data_save_path = os.path.join("dataset", name)
        self.word_to_id_path = os.path.join(self.data_save_path, "vocab.pkl")
        self.concept_to_id_path = os.path.join(self.data_save_path, "concept.pkl")

        self.word_emb_path = os.path.join(self.data_save_path, "word_embedding_glove.npz")
        self.concept_emb_path = os.path.join(self.data_save_path, "concept_embedding_glove.npz")

        self.word_concept_dic = json.loads(
            open("./concept/words_concepts_dict_top" + str(concept) + ".txt", "r").read())
        self.w_w_edge_id_path = os.path.join(self.data_save_path, "w_w_edge.pkl")
        self.w_c_edge_id_path = os.path.join(self.data_save_path, "w_c_edge.pkl")
        self.c_w_edge_id_path = os.path.join(self.data_save_path, "c_w_edge.pkl")
        self.w_w_edge_emb_path = os.path.join(self.data_save_path, "w_w_edge_embedding.npz")
        self.w_c_edge_emb_path = os.path.join(self.data_save_path, "w_c_edge_embedding.npz")
        self.c_w_edge_emb_path = os.path.join(self.data_save_path, "c_w_edge_embedding.npz")

        self.log_path = self.data_save_path
        self.model_path = os.path.join(self.data_save_path, "model.ckpt")

        self.graphs = []
        self.labels = []
        assert sub in ['train', 'dev', 'test']
        super(MyDataset, self).__init__(name=name)

    def process(self):
        # load word and concept to id
        word_to_id = pkl.load(open(self.word_to_id_path, 'rb'))
        id_to_word = {value: key for key, value in word_to_id.items()}

        concept_to_id = pkl.load(open(self.concept_to_id_path, 'rb'))
        id_to_concept = {value: key for key, value in concept_to_id.items()}

        # load w_w_edge to id
        w_w_edge_to_id = pkl.load(open(self.w_w_edge_id_path, 'rb'))
        id_to_w_w = {value: key for key, value in w_w_edge_to_id.items()}

        w_c_edge_to_id = pkl.load(open(self.w_c_edge_id_path, 'rb'))
        id_to_w_c = {value: key for key, value in w_c_edge_to_id.items()}

        c_w_edge_to_id = pkl.load(open(self.c_w_edge_id_path, 'rb'))
        id_to_c_w = {value: key for key, value in c_w_edge_to_id.items()}

        loss_data = 0

        if os.path.exists(os.path.join(self.data_save_path, 'dgl_graph_{}_{}.bin'.format(self.hash, self.sub))):
            self.load()
        else:
            files = ['{}.txt'.format(self.sub)]
            with open(os.path.join(self.data_save_path, files[0]), encoding='utf-8') as vf:
                for line in vf.readlines():
                    # print(line)
                    lin = line.strip()
                    if len(lin.split("\t")) != 2:
                        continue
                    label = lin.split('\t')[0]
                    content = lin.split('\t')[1]
                    content = nltk.word_tokenize(content)

                    content_ids = [word_to_id.get(c, word_to_id.get(UNK)) for c in content]
                    conten_len = len(content_ids)
                    word_word = []
                    word_concept = []
                    concept_word = []
                    for i in range(conten_len):
                        if content[i] in self.word_concept_dic.keys():
                            for k in self.word_concept_dic.get(content[i]).keys():
                                if k in concept_to_id.keys():
                                    word_concept.append((content_ids[i], concept_to_id[k]))
                                    concept_word.append((concept_to_id[k], content_ids[i]))
                        for j in range(i - self.neighbor, i + self.neighbor + 1):
                            # 超过边界跳过
                            if j < 0 or j >= conten_len:
                                continue
                            word_word.append((content_ids[i], content_ids[j]))

                    word_number_to_order_number = {}  # such as ： 5->0, 14->1
                    concept_number_to_order_number = {}  # such as ： 1->0, 5->1
                    for w in set(content_ids):
                        word_number_to_order_number[w] = len(word_number_to_order_number)

                    total_concept_ids = []
                    for tu in concept_word:
                        total_concept_ids.append(tu[0])

                    for w in set(total_concept_ids):
                        concept_number_to_order_number[w] = len(concept_number_to_order_number)

                    w_w_ids = []
                    w_A_A = []
                    w_A_B = []
                    for w_w in word_word:
                        a = word_number_to_order_number[w_w[0]]
                        b = word_number_to_order_number[w_w[1]]
                        w_A_A.append(a)
                        w_A_B.append(b)
                        w_w_ids.append(w_w_edge_to_id.get(id_to_word[w_w[0]] + "||" + id_to_word[w_w[1]]))

                    w_c_ids = []
                    w_B_A = []
                    w_B_B = []
                    for w_c in word_concept:
                        a = word_number_to_order_number[w_c[0]]
                        b = concept_number_to_order_number[w_c[1]]
                        w_B_A.append(a)
                        w_B_B.append(b)
                        w_c_ids.append(w_c_edge_to_id.get(id_to_word[w_c[0]] + "||" + id_to_concept[w_c[1]]))

                    c_w_ids = []
                    w_C_A = []
                    w_C_B = []
                    for c_w in concept_word:
                        a = concept_number_to_order_number[c_w[0]]
                        b = word_number_to_order_number[c_w[1]]
                        w_C_A.append(a)
                        w_C_B.append(b)
                        c_w_ids.append(c_w_edge_to_id.get(id_to_concept[c_w[0]] + "||" + id_to_word[c_w[1]]))

                    if len(w_B_A) == 0:
                        loss_data += 1
                        continue

                    graph_data = {
                        ('word', 'A', 'word'): (torch.tensor(w_A_A), torch.tensor(w_A_B)),
                        ('word', 'B', 'concept'): (torch.tensor(w_B_A), torch.tensor(w_B_B)),
                        ('concept', 'C', 'word'): (torch.tensor(w_C_A), torch.tensor(w_C_B))
                    }

                    g = dgl.heterograph(graph_data)
                    # order_number_to_word_number = {value: key for key, value in word_number_to_order_number.items()}
                    # order_number_to_concept_number = {value: key for key, value in concept_number_to_order_number.items()}

                    g.nodes['word'].data['x'] = torch.tensor(list(word_number_to_order_number.keys()))
                    g.nodes['concept'].data['x'] = torch.tensor(list(concept_number_to_order_number.keys()))

                    g.edges['A'].data['h'] = torch.tensor(w_w_ids)
                    g.edges['B'].data['h'] = torch.tensor(w_c_ids)
                    g.edges['C'].data['h'] = torch.tensor(c_w_ids)

                    edge_dict = {}
                    for etype in g.etypes:
                        edge_dict[etype] = len(edge_dict)
                        g.edges[etype].data['id'] = torch.ones(g.number_of_edges(etype), dtype=torch.long) * edge_dict[
                            etype]

                    self.graphs.append(g)
                    self.labels.append(int(label))

                    # print(int(label))
            # print("loss data:", loss_data)
            self.labels = torch.tensor(self.labels)
            self.save()

    @property
    def vocab_size(self):
        r"""Vocabulary size."""
        return len(self._vocab)

    @property
    def num_labels(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return self.class_num

    def __getitem__(self, idx):
        """ Get graph and label by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (dgl.DGLGraph, Tensor)
        """
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.graphs)

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.data_save_path, 'dgl_graph_{}_{}.bin'.format(self.hash, self.sub))
        save_graphs(str(graph_path), self.graphs, {'labels': self.labels})

    def load(self):
        graphs, label_dict = load_graphs(
            os.path.join(self.data_save_path, 'dgl_graph_{}_{}.bin'.format(self.hash, self.sub)))
        self.graphs = graphs
        self.labels = label_dict['labels']
