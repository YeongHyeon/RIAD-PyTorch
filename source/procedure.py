import os, time
import numpy as np
import pandas as pd
import source.utils as utils

def training(agent, dataset, batch_size, epochs):

    savedir = 'results_tr'
    utils.make_dir(path=savedir, refresh=True)
    utils.make_dir(path=os.path.join(savedir, 'gen'), refresh=True)
    utils.make_dir(path=os.path.join(savedir, 'dic'), refresh=True)

    iter_per_epoch = 0
    while(True):
        minibatch = dataset.next_batch(batch_size=batch_size, ttv=0)
        if(minibatch['x'].shape[0] == 0): break
        iter_per_epoch += 1
        if(minibatch['terminate']): break

    print("\n** Training to %d epoch | Batch size: %d" %(epochs, batch_size))
    iteration = 0
    dict_best = { \
        'best_loss': 1e+30, 'best_loss_ep': 0, \
        'best_auroc': 0, 'best_auroc_ep': 0}
    dic_history = {'epoch':[], 'time':[], 'loss':[], 'auroc':[]}

    for epoch in range(epochs):

        step_dict = agent.step(minibatch=dataset.batchviz, training=False)
        utils.plot_generation( \
            step_dict['y'], step_dict['y_hat'], step_dict['map'], \
            savepath=os.path.join(savedir, 'gen', 'generation_%08d.png' %(epoch)))
        utils.save_pkl(\
            path=os.path.join(savedir, 'dic', 'dic_%08d.pkl' %(epoch)), pkl=step_dict)

        list_loss = []
        while(True):
            minibatch = dataset.next_batch(batch_size=batch_size, ttv=0)

            if(minibatch['x'].shape[0] == 0): break
            step_dict = agent.step(minibatch=minibatch, \
                iteration=iteration, epoch=epoch, iter_per_epoch=iter_per_epoch, training=True)
            list_loss.append(step_dict['losses']['l2'])
            iteration += 1

            if(minibatch['terminate']): break

        loss_tmp = np.average(np.asarray(list_loss))

        dic_measure = {'label':[], 'score':[]}
        while(True):
            minibatch = dataset.next_batch(batch_size=batch_size, ttv=2)
            if(minibatch['x'].shape[0] == 0): break
            step_dict = agent.step(minibatch=minibatch, training=False)

            dic_measure['label'].extend(list(minibatch['y']))
            dic_measure['score'].extend(list(step_dict['losses']['l2_b']))
            if(minibatch['terminate']): break

        auroc_tmp = utils.measure_auroc( \
            dic_measure['label'], dic_measure['score'], \
            savepath=os.path.join(savedir, 'gen', 'auroc_%08d.png' %(epoch)))

        dic_history['epoch'].append(epoch)
        dic_history['time'].append(time.time())
        dic_history['loss'].append(loss_tmp)
        dic_history['auroc'].append(auroc_tmp)

        if(dict_best['best_loss'] >= loss_tmp):
            dict_best['best_loss'] = loss_tmp
            dict_best['best_loss_ep'] = epoch
            agent.save_params(model='model_1_best_loss')
        if(dict_best['best_auroc'] <= auroc_tmp):
            dict_best['best_auroc'] = auroc_tmp
            dict_best['best_auroc_ep'] = epoch
            agent.save_params(model='model_2_best_auroc')
        agent.save_params(model='model_0_finepoch')

        print("Epoch [%d / %d] | Loss: %f (best at %d epoch)  AUROC: %f (best at %d epoch)" \
            %(epoch, epochs, loss_tmp, dict_best['best_loss_ep'], auroc_tmp, dict_best['best_auroc_ep']))

    return dict_best

def test(agent, dataset):

    batch_size = 1
    savedir = 'results_te'
    utils.make_dir(path=savedir, refresh=True)

    list_model = utils.sorted_list(os.path.join('Checkpoint', 'model*.pth'))
    for idx_model, path_model in enumerate(list_model):
        list_model[idx_model] = path_model.split('/')[-1]

    dict_best = {'name_best': '', 'auroc': 0, 'loss': 0}

    for idx_model, path_model in enumerate(list_model):

        print("\n** Test with %s" %(path_model))
        try:
            agent.load_params(model=path_model)
            name_model = path_model.replace('.pth', '')
            utils.make_dir(path=os.path.join(savedir, name_model), refresh=False)
        except: continue

        dic_measure = {'label':[], 'score':[]}
        while(True):
            minibatch = dataset.next_batch(batch_size=batch_size, ttv=2)
            if(minibatch['x'].shape[0] == 0): break
            step_dict = agent.step(minibatch=minibatch, training=False)

            dic_measure['label'].extend(list(minibatch['y']))
            dic_measure['score'].extend(list(step_dict['losses']['l2_b']))
            if(minibatch['terminate']): break

        auroc_tmp = utils.measure_auroc(dic_measure['label'], dic_measure['score'], savepath=None)

        dic_score = {'label':dic_measure['label'], 'score':dic_measure['score']}
        df_score = pd.DataFrame.from_dict(dic_score)
        df_score.to_csv(os.path.join(savedir, "test_%s.csv" %(name_model)), index=False)

        if(dict_best['auroc'] < auroc_tmp):
            dict_best['name_best'] = path_model
            dict_best['auroc'] = float(auroc_tmp)
            dict_best['loss'] = float(np.average(list(df_score.loc[df_score['label'] == 0]['score'])))

    return dict_best, len(list_model)
