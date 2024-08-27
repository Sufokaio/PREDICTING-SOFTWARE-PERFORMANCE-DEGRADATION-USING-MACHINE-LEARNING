import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=RuntimeWarning)
import numpy as np
import tensorflow as tf
import random as rd
from time import time

from util.setting import init_setting, log
from util.Word2vec import Word2vec

from model.load_gnn import GNNLoader
from model.SGL import SGL
from model.eval import perf_val_supervised, perf_test_supervised

def main():
    """ Get argument settings """
    seed = 2022
    np.random.seed(seed)
    rd.seed(seed)
    tf.set_random_seed(seed)

    """ Initialize args and dataset """
    
    args = init_setting()
    

    
    log.info("Loading data from %s" % args.dataset)
    data_generator = GNNLoader(args)
    
    """ Use pre-trained word2vec embeddings to initialize AST nodes (entities) """
    type2vec = Word2vec(args)
    if args.pretrain == 0:
        type2vec.init_embedding(data_generator.e2t_list, data_generator.typetoken_seq)
        """ Save pre-trained word2vec embeddings"""
        if args.save_model:
            type2vec.store_embedding(data_generator.out_path)
    elif args.pretrain == 1:
        type2vec.load_embedding(data_generator.out_path)

    # different types of degradation
    for degradation_type in [0]:
        log.info("code performance degradation prediction on type %d", degradation_type)
        data_generator._generate_perf_c4b_split(degradation_type)

        """ Select learning models """
        if args.model_type == 'oaktree':
            log.info("Initing SGL model")
            model = SGL(args, data_generator, pretrain_embedding=type2vec.embedding)
        else:
            log.error("The ML model is unknown")
            exit(-1)

        """ Setup tensorflow session """
        log.info("Setup tensorflow session")
        sess = model.setup_sess()
        
        """ Reload model parameters for fine tune """
        if args.pretrain == 2:
            model.load_model(sess, data_generator.out_path)

        """ Training phase """
        log.info("Training %d for %d epochs", degradation_type, args.epoch)
        for epoch in range(args.epoch):
            model.lr = 0.1
            if epoch > 10:
                model.lr = 0.01

            perf_loss = 0.
            t_train = time()
            
            """ fine-tune for code performance degradation (supervised) """
            if args.perf_test_supervised:
                rd.shuffle(data_generator.perf_train_data)
                for i_batch in range(data_generator.perf_data_iter):
                    batch_data = data_generator.generate_perf_train_batch(i_batch)
                    feed_dict = data_generator.generate_perf_train_feed_dict(model, batch_data)
                    _, perf_loss_batch = model.train_perf(sess, feed_dict)
                    perf_loss += perf_loss_batch
                perf_train_ite = 'Epoch %d [%.1fs]: train=[(performance degradation: %.5f)]' % (epoch + 1, time() - t_train, perf_loss)

                if np.isnan(perf_loss) == True:
                    log.error('error: loss@ is nan')
                    exit(-1)

            log.debug(perf_train_ite)

            if epoch % 1 == 0:
                """ perf test supervised """
                if args.perf_test_supervised:
                    perf_val_supervised(sess, model, data_generator, args.degradation_threshold)    

        """ Testing phase """
        if args.perf_test_supervised:
            perf_test_supervised(sess, model, data_generator, args.degradation_threshold)

    """ Save the model parameters """
    if args.save_model:
        model.store_model(sess, data_generator.out_path, epoch)

if __name__ == '__main__':
    main()
