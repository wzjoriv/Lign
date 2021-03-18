from lign.train import norm_labels
from lign.utils import clustering as cl
import torch as th
from lign.utils import io

import matplotlib.pyplot as plt

def validate(model, graph, train, tag_in, tag_out, labels, metrics = ['accuracy'], sv_img = None, cluster = (cl.NN(), 3), device = th.device('cpu')):

    model.eval()
    with th.no_grad():
        tr_nodes, tr_labs = cl.filter_k(tag_out, labels, train, cluster[1])
        sub = train.subgraph(tr_nodes)
        inp = sub.get_parent_data(tag_in).to(device)
        
        cluster = cluster[0]
        cluster.train(model(sub, inp), tr_labs.to(device))

        ts_nodes = cl.filter(tag_out, labels, graph)
        graph = graph.subgraph(ts_nodes)

        inp = graph.get_parent_data(tag_in).to(device)
        outp_t = graph.get_parent_data(tag_out).to(device)

        rep_vec = model(graph, inp)
        outp_p = cluster(rep_vec)

        print("Prediction: ")
        print(outp_p[:6])

        print("True values: ")
        print(outp_t[:6])

        print("Vector output: ")
        print(rep_vec[:6])


    ## save 2d image
    if sv_img is not None and sv_img == '2d':

        fig = plt.figure()

        tp = rep_vec.cpu().detach().numpy()
        tp2 = outp_t.cpu().detach().numpy()
        c = plt.scatter(tp[:, 0], tp[:, 1], c=tp2, cmap=plt.get_cmap('gist_rainbow'))
        #plt.ylim([-1.2,1.2])
        #plt.xlim([-1.2,1.2])

        plt.colorbar(c).set_label("Label")
        plt.savefig("data/views-2d/Validate "+str(len(labels))+".png")
        plt.close()


    elif sv_img != None and sv_img == '3d':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        tp = rep_vec.cpu().detach().numpy()
        tp2 = outp_t.cpu().detach().numpy()
        c = ax.scatter(tp[:, 0], tp[:, 1], tp[:, 2], c=tp2, cmap=plt.get_cmap('gist_rainbow'))

        plt.colorbar(c).set_label("Label")
        plt.savefig("data/views-3d/Validate "+str(len(labels))+".png")
        plt.close()

    out = {}
    metrics = io.to_iter(metrics)
    for metric in metrics:
        if metric == 'accuracy':
            out[metric] = (outp_p == outp_t).sum().item() * 100.0/outp_p.size(0)
    
    return out

def accuracy(model, graph, train, tag_in, tag_out, labels, cluster = (cl.NN(), 3), sv_img = None, device = th.device('cpu')):

    out = validate(model, graph, train, tag_in, tag_out, labels, metrics = 'accuracy', cluster = cluster, sv_img=sv_img, device = device)

    return out['accuracy']