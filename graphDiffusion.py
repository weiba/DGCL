from torch_geometric.data import Data
import torch_geometric.transforms as T


def generate_auxiliary_graph(args, data):
    """
    Construct the auxiliary graph based on the original graph.
    :param args: Arguments received from command line
    :param dataset (class: dictionary): Dataset loaded from a pickle file
    :return (class: torch.LongTensor): edges indexed by ids
    """
    data = Data(x=data.x, y=data.y, edge_index=data.edge_index)
    if args.graph_diffusion == 'ppr':
        gdc = T.GDC(self_loop_weight=None, normalization_in='sym',
                    normalization_out='col',
                    diffusion_kwargs=dict(method=args.graph_diffusion, alpha=args.ppr_alpha, eps=args.ppr_eps),
                    sparsification_kwargs=dict(method='threshold', avg_degree=args.net_avg_deg),
                    exact=True)
        data = gdc(data)
        print('diffusion graph of %s is finished...' % args.graph_diffusion)
        return data.edge_index

    elif args.graph_diffusion == 'heat':
        gdc = T.GDC(self_loop_weight=None, normalization_in='sym',
                    normalization_out='col',
                    diffusion_kwargs=dict(method=args.graph_diffusion, t=args.hk_t),
                    sparsification_kwargs=dict(method='threshold', avg_degree=args.net_avg_deg),
                    exact=True)
        data = gdc(data)
        print('diffusion graph of %s is finished...' % args.graph_diffusion)
        return data.edge_index




