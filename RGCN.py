import torch
from dgl.nn.pytorch.utils import Identity
from dgl.ops import edge_softmax
from torch import nn
import dgl
import torch.nn.functional as F
import numpy as np
from torch.nn import init
import dgl.function as fn
from dgl import DGLError
from dgl.utils import expand_as_pair, check_eq_shape


class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, layer):
        super().__init__()

        self.conv1 = dgl.nn.HeteroGraphConv({
            rel: GraphConv(in_feats, out_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dgl.nn.HeteroGraphConv({
            rel: GraphConv(in_feats, out_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv3 = dgl.nn.HeteroGraphConv({
            rel: GraphConv(in_feats, out_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv4 = dgl.nn.HeteroGraphConv({
            rel: GraphConv(in_feats, out_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv5 = dgl.nn.HeteroGraphConv({
            rel: GraphConv(in_feats, out_feats)
            for rel in rel_names}, aggregate='sum')
        self.layer = layer

    def forward(self, graph, inputs):
        # inputs are features of nodes
        if self.layer == 1:
            h = self.conv1(graph, inputs)
            h = {k: F.relu(v) for k, v in h.items()}
        elif self.layer == 2:
            h = self.conv1(graph, inputs)
            h = {k: F.relu(v) for k, v in h.items()}
            h = self.conv2(graph, h)
            h = {k: F.relu(v) for k, v in h.items()}
        elif self.layer == 3:
            h = self.conv1(graph, inputs)
            h = {k: F.relu(v) for k, v in h.items()}
            h = self.conv2(graph, h)
            h = {k: F.relu(v) for k, v in h.items()}
            h = self.conv3(graph, h)
            h = {k: F.relu(v) for k, v in h.items()}
        elif self.layer == 4:
            h = self.conv1(graph, inputs)
            h = {k: F.relu(v) for k, v in h.items()}
            h = self.conv2(graph, h)
            h = {k: F.relu(v) for k, v in h.items()}
            h = self.conv3(graph, h)
            h = {k: F.relu(v) for k, v in h.items()}
            h = self.conv4(graph, h)
            h = {k: F.relu(v) for k, v in h.items()}
        elif self.layer == 5:
            h = self.conv1(graph, inputs)
            h = {k: F.relu(v) for k, v in h.items()}
            h = self.conv2(graph, h)
            h = {k: F.relu(v) for k, v in h.items()}
            h = self.conv3(graph, h)
            h = {k: F.relu(v) for k, v in h.items()}
            h = self.conv4(graph, h)
            h = {k: F.relu(v) for k, v in h.items()}
            h = self.conv5(graph, h)
            h = {k: F.relu(v) for k, v in h.items()}
        return h


class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names, dataset, layer = 1, dropout=0.5):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.concept_embedding = nn.Embedding.from_pretrained(
            torch.tensor(
                np.load('dataset/' + dataset + '/concept_embedding_glove.npz')["embeddings"].astype('float32')),
            freeze=False
        )

        # print(self.concept_embedding.num_embeddings)

        self.word_embedding = nn.Embedding.from_pretrained(
            torch.tensor(np.load('dataset/' + dataset + '/word_embedding_glove.npz')["embeddings"].astype('float32')),
            freeze=False
        )
        self.w_w_embedding = nn.Embedding.from_pretrained(
            torch.tensor(np.load('dataset/' + dataset + '/w_w_edge_embedding.npz')["embeddings"].astype('float32')),
            freeze=False
        )
        self.w_c_embedding = nn.Embedding.from_pretrained(
            torch.tensor(np.load('dataset/' + dataset + '/w_c_edge_embedding.npz')["embeddings"].astype('float32')),
            freeze=False
        )
        self.c_w_embedding = nn.Embedding.from_pretrained(
            torch.tensor(np.load('dataset/' + dataset + '/c_w_edge_embedding.npz')["embeddings"].astype('float32')),
            freeze=False
        )

        self.W_word = nn.Parameter(torch.FloatTensor(in_dim, in_dim))
        self.W_concept = nn.Parameter(torch.FloatTensor(in_dim, in_dim))

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names , layer=layer)
        self.classify = nn.Linear(2 * hidden_dim, n_classes)

    def forward(self, g):
        g.nodes['word'].data['feat'] = self.dropout(self.word_embedding(g.nodes['word'].data['x']))
        g.nodes['concept'].data['feat'] = self.dropout(self.concept_embedding(g.nodes['concept'].data['x']))
        g.edges['A'].data['weight'] = self.dropout(self.w_w_embedding(g.edges['A'].data['h']))
        g.edges['B'].data['weight'] = self.dropout(self.w_c_embedding(g.edges['B'].data['h']))
        g.edges['C'].data['weight'] = self.dropout(self.c_w_embedding(g.edges['C'].data['h']))

        # g.nodes['word'].data['feat'] = self.word_embedding(g.nodes['word'].data['x'])
        # g.nodes['concept'].data['feat'] = self.concept_embedding(g.nodes['concept'].data['x'])
        # g.edges['A'].data['weight'] = self.w_w_embedding(g.edges['A'].data['h'])
        # g.edges['B'].data['weight'] = self.w_c_embedding(g.edges['B'].data['h'])
        # g.edges['C'].data['weight'] = self.c_w_embedding(g.edges['C'].data['h'])

        h = g.ndata['feat']
        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            # hg = 0
            hg = torch.cat(
                (
                    dgl.sum_nodes(g, 'h', ntype='word'),
                    dgl.sum_nodes(g, 'h', ntype='concept')
                ),
                -1
            )
            # for ntype in g.ntypes:
            #     hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
            #     r = torch.cat(dgl.sum_nodes(g, 'h', ntype=ntype))
            return self.classify(hg)


class GraphConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GraphConv, self).__init__()
        if norm not in ('none', 'both', 'right'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        if weight:
            self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        r"""
        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized as in the
        `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__
        where the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
        and the bias is initialized to be zero.
        """
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        r"""
        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, weight=None):
        r"""
        Description
        -----------
        Compute graph convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, which is the case for bipartite graph, the pair
            must contain two tensors of shape :math:`(N_{in}, D_{in_{src}})` and
            :math:`(N_{out}, D_{in_{dst}})`.
        eweight : torch.Tensor of shape (E, 1)
            Edge weights, E for the number of edges.
        weight : torch.Tensor, optional
            Optional external weight tensor.

        Returns
        -------
        torch.Tensor
            The output feature

        Raises
        ------
        DGLError
            Case 1:
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
            Case 2:
            External weight is provided while at the same time the module
            has defined its own weight parameter.

        Note
        ----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: :math:`(\text{in_feats}, \text{out_feats})`.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight

            # Set edge weights
            # graph.edata['w'] = eweight
            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = torch.matmul(feat_src, weight)
                graph.srcdata['h'] = feat_src
                # Changed from fn.copy_src to fn.u_mul_e
                graph.update_all(fn.u_mul_e(lhs_field='h', rhs_field='weight', out='m'),
                                 fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                # Changed from fn.copy_src to fn.u_mul_e
                graph.update_all(fn.u_mul_e(lhs_field='h', rhs_field='weight', out='m'),
                                 fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = torch.matmul(rst, weight)

            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class GatedGraphConv(nn.Module):
    r"""

    Description
    -----------
    Gated Graph Convolution layer from paper `Gated Graph Sequence
    Neural Networks <https://arxiv.org/pdf/1511.05493.pdf>`__.

    .. math::
        h_{i}^{0} &= [ x_i \| \mathbf{0} ]

        a_{i}^{t} &= \sum_{j\in\mathcal{N}(i)} W_{e_{ij}} h_{j}^{t}

        h_{i}^{t+1} &= \mathrm{GRU}(a_{i}^{t}, h_{i}^{t})

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`x_i`.
    out_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(t+1)}`.
    n_steps : int
        Number of recurrent steps; i.e, the :math:`t` in the above formula.
    n_etypes : int
        Number of edge types.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GatedGraphConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> conv = GatedGraphConv(10, 10, 2, 3)
    >>> etype = th.tensor([0,1,2,0,1,2])
    >>> res = conv(g, feat, etype)
    >>> res
    tensor([[ 0.4652,  0.4458,  0.5169,  0.4126,  0.4847,  0.2303,  0.2757,  0.7721,
            0.0523,  0.0857],
            [ 0.0832,  0.1388, -0.5643,  0.7053, -0.2524, -0.3847,  0.7587,  0.8245,
            0.9315,  0.4063],
            [ 0.6340,  0.4096,  0.7692,  0.2125,  0.2106,  0.4542, -0.0580,  0.3364,
            -0.1376,  0.4948],
            [ 0.5551,  0.7946,  0.6220,  0.8058,  0.5711,  0.3063, -0.5454,  0.2272,
            -0.6931, -0.1607],
            [ 0.2644,  0.2469, -0.6143,  0.6008, -0.1516, -0.3781,  0.5878,  0.7993,
            0.9241,  0.1835],
            [ 0.6393,  0.3447,  0.3893,  0.4279,  0.3342,  0.3809,  0.0406,  0.5030,
            0.1342,  0.0425]], grad_fn=<AddBackward0>)
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 n_steps,
                 n_etypes,
                 bias=True):
        super(GatedGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._n_steps = n_steps
        self._n_etypes = n_etypes
        self.linears = nn.ModuleList(
            [nn.Linear(out_feats, out_feats) for _ in range(n_etypes)]
        )
        self.gru = nn.GRUCell(out_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized using Glorot uniform initialization
        and the bias is initialized to be zero.
        """
        gain = init.calculate_gain('relu')
        self.gru.reset_parameters()
        for linear in self.linears:
            init.xavier_normal_(linear.weight, gain=gain)
            init.zeros_(linear.bias)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, etypes):
        """

        Description
        -----------
        Compute Gated Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`N`
            is the number of nodes of the graph and :math:`D_{in}` is the
            input feature size.
        etypes : torch.LongTensor
            The edge type tensor of shape :math:`(E,)` where :math:`E` is
            the number of edges of the graph.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is the output feature size.
        """
        with graph.local_scope():
            assert graph.is_homogeneous, \
                "not a homogeneous graph; convert it with to_homogeneous " \
                "and pass in the edge type as argument"
            assert etypes.min() >= 0 and etypes.max() < self._n_etypes, \
                "edge type indices out of range [0, {})".format(self._n_etypes)
            zero_pad = feat.new_zeros((feat.shape[0], self._out_feats - feat.shape[1]))
            feat = torch.cat([feat, zero_pad], -1)

            for _ in range(self._n_steps):
                graph.ndata['h'] = feat
                for i in range(self._n_etypes):
                    eids = (etypes == i).nonzero().view(-1).type(graph.idtype)
                    if len(eids) > 0:
                        graph.apply_edges(
                            lambda edges: {'W_e*h': self.linears[i](edges.src['h'])},
                            eids
                        )
                graph.update_all(fn.copy_e('W_e*h', 'm'), fn.sum('m', 'a'))
                a = graph.ndata.pop('a') # (N, D)
                feat = self.gru(a, feat)
            return feat