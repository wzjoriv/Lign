
import torch as th
from lign.utils import clustering as cl
from lign.utils import io
from lign.utils import functions as fn

import matplotlib.pyplot as plt
import numpy as np

def g_validate(model, graph, train, tag_in, tag_out, labels, metrics = ['accuracy'], sv_img = None, cluster = (cl.NN(), 3), device = th.device('cpu'), save_img = False):

    model.eval()
    with th.no_grad():
        tr_nodes, tr_labs = fn.filter_k_from_tags(tag_out, labels, train, cluster[1])
        sub = train.subgraph(tr_nodes)
        inp = sub.get_parent_data(tag_in).to(device)
        
        cluster = cluster[0]
        cluster.train(model(sub, inp), tr_labs.to(device))

        ts_nodes = fn.filter_tags(tag_out, labels, graph)
        graph = graph.subgraph(ts_nodes)

        inp = graph.get_parent_data(tag_in).to(device)
        outp_t = graph.get_parent_data(tag_out).to(device)

        rep_vec = model(graph, inp, save=save_img)
        outp_p = cluster(rep_vec)


    ## save 2d image
    if sv_img is not None:
        if sv_img == '2d':

            fig = plt.figure()

            tp = rep_vec.cpu().detach().numpy()
            tp2 = outp_t.cpu().detach().numpy()
            c = plt.scatter(tp[:, 0], tp[:, 1], c=tp2, cmap=plt.get_cmap('gist_rainbow'))
            #plt.ylim([-1.2,1.2])
            #plt.xlim([-1.2,1.2])

            plt.colorbar(c).set_label("Label")
            plt.savefig("data/views-2d/Validate "+str(len(labels))+".png")
            plt.close()


        elif sv_img == '3d':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            tp = rep_vec.cpu().detach().numpy()
            tp2 = outp_t.cpu().detach().numpy()
            c = ax.scatter(tp[:, 2], tp[:, 3], tp[:, 4], c=tp2, cmap=plt.get_cmap('gist_rainbow'))

            plt.colorbar(c).set_label("Label")
            plt.savefig("data/views-3d/Validate "+str(len(labels))+".png")
            plt.close()


        elif sv_img == 'bt':
            subset = 20

            inp_tp = inp.cpu().detach().numpy()
            tp = rep_vec.cpu().detach().numpy()
            tp2 = outp_t.cpu().detach().numpy()
            tp3 = outp_p.cpu().detach().numpy()

            para = {
                "PREDICTION": tp3[:subset].tolist(),
                "TRUTH": tp2[:subset].tolist(),
                "VECTOR": tp[:subset].tolist(),
            }

            io.json(para, "data/views-bt/Validate "+str(len(labels))+".json")
            
            for i in range(subset):
                plt.imshow(inp_tp[i][0], interpolation='nearest')
                plt.savefig("data/views-bt/imgs/Validate "+str(len(labels))+ "-" + str(i) + ".png")

            plt.close()

    out = {}
    metrics = io.to_iter(metrics)
    for metric in metrics:
        if metric == 'accuracy':
            out[metric] = (outp_p == outp_t).sum().item() * 100.0/outp_p.size(0)
    
    return out

def validate(model, graph, train, tag_in, tag_out, labels, metrics = ['accuracy'], sv_img = None, cluster = (cl.NN(), 3), device = th.device('cpu'), save_img = False):

    model.eval()
    with th.no_grad():
        tr_nodes, tr_labs = fn.filter_k_from_tags(tag_out, labels, train, cluster[1])
        sub = train.subgraph(tr_nodes)
        inp = sub.get_parent_data(tag_in).to(device)
        
        cluster = cluster[0]
        cluster.train(model(sub, inp), tr_labs.to(device))

        ts_nodes = fn.filter_tags(tag_out, labels, graph)
        graph = graph.subgraph(ts_nodes)

        inp = graph.get_parent_data(tag_in).to(device)
        outp_t = graph.get_parent_data(tag_out).to(device)

        rep_vec = model(graph, inp, save=save_img)
        outp_p = cluster(rep_vec)


    ## save 2d image
    if sv_img is not None:
        if sv_img == '2d':

            fig = plt.figure()

            tp = rep_vec.cpu().detach().numpy()
            tp2 = outp_t.cpu().detach().numpy()
            c = plt.scatter(tp[:, 0], tp[:, 1], c=tp2, cmap=plt.get_cmap('gist_rainbow'))
            #plt.ylim([-1.2,1.2])
            #plt.xlim([-1.2,1.2])

            plt.colorbar(c).set_label("Label")
            plt.savefig("data/views-2d/Validate "+str(len(labels))+".png")
            plt.close()


        elif sv_img == '3d':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            tp = rep_vec.cpu().detach().numpy()
            tp2 = outp_t.cpu().detach().numpy()
            c = ax.scatter(tp[:, 2], tp[:, 3], tp[:, 4], c=tp2, cmap=plt.get_cmap('gist_rainbow'))

            plt.colorbar(c).set_label("Label")
            plt.savefig("data/views-3d/Validate "+str(len(labels))+".png")
            plt.close()


        elif sv_img == 'bt':
            subset = 20

            inp_tp = inp.cpu().detach().numpy()
            tp = rep_vec.cpu().detach().numpy()
            tp2 = outp_t.cpu().detach().numpy()
            tp3 = outp_p.cpu().detach().numpy()

            para = {
                "PREDICTION": tp3[:subset].tolist(),
                "TRUTH": tp2[:subset].tolist(),
                "VECTOR": tp[:subset].tolist(),
            }

            io.json(para, "data/views-bt/Validate "+str(len(labels))+".json")
            
            for i in range(subset):
                plt.imshow(inp_tp[i][0], interpolation='nearest')
                plt.savefig("data/views-bt/imgs/Validate "+str(len(labels))+ "-" + str(i) + ".png")

            plt.close()

    out = {}
    metrics = io.to_iter(metrics)
    for metric in metrics:
        if metric == 'accuracy':
            out[metric] = (outp_p == outp_t).sum().item() * 100.0/outp_p.size(0)
    
    return out

def accuracy(model, graph, train, tag_in, tag_out, labels, cluster = (cl.NN(), 3), sv_img = None, device = th.device('cpu'), save_img = False):

    out = validate(model, graph, train, tag_in, tag_out, labels, metrics = 'accuracy', cluster = cluster, sv_img=sv_img, device = device, save_img = save_img)

    return out['accuracy']