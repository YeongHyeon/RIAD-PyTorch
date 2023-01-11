import argparse, time, os, operator

import torch
import numpy as np
import source.agent as agt
import source.utils as utils
import source.procedure as proc
import source.datamanager as dman

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu

    ngpu = FLAGS.ngpu
    if(not(torch.cuda.is_available())): ngpu = 0
    device = torch.device("cuda" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    dataset = dman.DataSet( \
        select_norm=FLAGS.select_norm, masking_mode=FLAGS.masking_mode, \
        disjoint_n=FLAGS.disjoint_n)

    agent = agt.Agent( \
        dim_h=dataset.dim_h, dim_w=dataset.dim_w, dim_c=dataset.dim_c, \
        nn=FLAGS.nn, ksize=FLAGS.ksize, learning_rate=FLAGS.lr, mode_lr=FLAGS.mode_lr, \
        mode_optim=FLAGS.mode_optim, path_ckpt='Checkpoint', ngpu=ngpu, device=device)

    time_tr = time.time()
    dict_train = proc.training( \
        agent=agent, dataset=dataset, \
        batch_size=FLAGS.batch, epochs=FLAGS.epochs)
    time_te = time.time()

    dict_best, num_model = proc.test( \
        agent=agent, dataset=dataset)
    time_fin = time.time()

    tr_time = time_te - time_tr
    te_time = time_fin - time_te

    dict_best.update(dataset.config)
    dict_best.update(agent.config)
    utils.save_json('result.json', dict_best)

    print("Time (TR): %.5f [sec]" \
        %(tr_time))
    print("Time (TE): %.5f (%.5f [sec/sample])" \
        %(te_time, te_time/num_model/dataset.num_te))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default="0", help='')
    parser.add_argument('--ngpu', type=int, default=1, help='')

    parser.add_argument('--select_norm', type=int, default=1, help='')
    parser.add_argument('--masking_mode', type=str, default='disjoint_mask', help='')
    parser.add_argument('--disjoint_n', type=int, default=3, help='')

    parser.add_argument('--nn', type=int, default=2000, help='')
    parser.add_argument('--ksize', type=int, default=3, help='')
    parser.add_argument('--lr', type=float, default=5e-4, help='')
    parser.add_argument('--mode_lr', type=int, default=0, help='')
    parser.add_argument('--mode_optim', type=str, default='sgd', help='')

    parser.add_argument('--batch', type=int, default=16, help='')
    parser.add_argument('--epochs', type=int, default=300, help='')

    FLAGS, unparsed = parser.parse_known_args()

    main()
