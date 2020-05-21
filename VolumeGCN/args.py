import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Volume-GCN")

    # training scheme
    parser.add_argument('--epochs', default=100, type=int)
    # parser.add_argument('--batch_size_con', default=10000, type=int)
    # parser.add_argument('--batch_size', default=5000, type=int)
    # parser.add_argument('--eval_size', default=3000, type=int)

    parser.add_argument('--lr', default=0.01, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=100, type=int)

    # model
    parser.add_argument('--dim', type=int, default=256,
                        help='Number of dimensions. Default is 256  (0-255).')
    parser.add_argument('--labeled_node', type=int, default=100,
                        help='Number of labeled nodes.')
    parser.add_argument('--label', type=int, default=1,
                        help='Number of label types.')
    # parser.add_argument('--k', type=int, default=5,
    #                     help='Subgraph size. Default is 5.')
    parser.add_argument('--dataset', default='datasets/jet_mixfrac_0051_supervoxels.gexf',
                        help='Name of dataset')
    parser.add_argument('--node_num', type=int, default=None,
                        help='Number of nodes.')

    args = parser.parse_args()
    return args
