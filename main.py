from __future__ import print_function
import os
import time
import argparse
import tensorflow.compat.v1 as tf
from sampler import WarpSampler
from model import Model
from tqdm import tqdm
from util import *

tf.disable_v2_behavior()
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
print('GPU devices:')
print(physical_devices)
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def str2list(v):
    return [int(i) for i in v.split(',')]
    
def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name, action='store_true')
    group.add_argument('--no-' + name, dest=name, action='store_false')
    parser.set_defaults(**{name:default})

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True)
parser.add_argument('--model', default='SASRec', type=str)
parser.add_argument('--model_dir', default='./tmp_model/', type=str)
parser.add_argument('--main_loss', default='point', type=str)
parser.add_argument('--int_match_loss', default='point', type=str)
parser.add_argument('--mode', default='causal', type=str)
parser.add_argument('--train_dir', required=True)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--temper', default=2, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.999, type=float)
parser.add_argument('--maxlen', default=50, type=int)
parser.add_argument('--mintestlen', default=12, type=int)
parser.add_argument('--hidden_units', default=50, type=int)
parser.add_argument('--u_hidden_units', default=4, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=201, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--eval_interval', default=20, type=int)
parser.add_argument('--eval_k', default=[10,5,20], type=str2list)
parser.add_argument('--next_it_rec_dilation', default=[1,2,4,8,1,2,4,8], type=str2list)
parser.add_argument('--next_it_rec_kernel_size', default=3, type=int)
parser.add_argument('--dropout_rate', default=0.5, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--norm', default=True, type=str2bool)
parser.add_argument('--debias', default=False, type=str2bool)
parser.add_argument('--ortho_loss_w', default=0.5, type=float)
parser.add_argument('--pop_loss_w', default=0.02, type=float)
parser.add_argument('--pop_match_loss_w', default=0.02, type=float)
parser.add_argument('--int_match_loss_w', default=0.02, type=float)
parser.add_argument('--c0', default=0.0, type=float)
parser.add_argument('--disentangle', default=False, type=str2bool)
parser.add_argument('--pop_match_tower', default=True, type=str2bool)
parser.add_argument('--dynamic_pop_int_weight', default=False, type=str2bool)
parser.add_argument('--enable_u', default=0, type=int)
parser.add_argument('--backbone', default=0, type=int)
parser.add_argument('--additive_bias', default=True, type=str2bool)
parser.add_argument('--pda_bias', default=False, type=str2bool)
parser.add_argument('--ipw_min', default=0.1, type=float)
parser.add_argument('--ipw_factor', default=0.5, type=float)

def init_model(session, saver, args):
    if args.model_dir.find('tmp_model') == -1:
        ckpt = tf.train.get_checkpoint_state(args.model_dir)
    else:
        ckpt = False
        
    if ckpt:
        print("restore all parameters")
        saver.restore(session, ckpt.model_checkpoint_path)
        return True
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        return False

args = parser.parse_args()
#if not os.path.isdir(args.dataset + '_' + args.train_dir):
#    os.makedirs(args.dataset + '_' + args.train_dir)
#with open(os.path.join(args.dataset + '_' + args.train_dir, 'args.txt'), 'w') as f:
#    f.write('\n'.join([str(k) + ',' + str(v) for k, v in sorted(vars(args).items(), key=lambda x: x[0])]))
#f.close()

if __name__ == '__main__':
    dataset = data_partition(args.dataset, args.mintestlen)
    [user_train, user_valid, user_test, usernum, itemnum, User, Item, item_prop] = dataset
    num_batch = len(user_train) // args.batch_size
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print ('average sequence length: %.2f, user count: %d, item count: %d' % (cc / len(user_train), usernum, itemnum))

    cc = 0.0
    test_u = 0
    for u in user_valid:
        if len(user_valid[u]) > 0:
            test_u += 1
            cc += len(user_train[u])
    print ('average sequence length with test: %.2f, test user count: %d' % (cc / test_u, test_u))

    total_interactions = 0
    user_int_cnt = []
    for u in User:
        user_int_cnt.append(len(User[u]))
        total_interactions += len(User[u])
    print('user gini index:%.4f' % gini(user_int_cnt))

    item_int_cnt = []
    for i in Item:
        item_int_cnt.append(len(Item[i]))
    print('item gini index:%.4f' % gini(item_int_cnt))

    prop0 = 0.0
    prop1 = 0.0
    for i in Item:
        if item_prop[i] < 0.01:
            prop0 += 1
        
        if item_prop[i] < 0.05:
            prop1 += 1  
            
    print('total interactions:%d; original propensity <0.01 ratio:%.4f, <0.05 ratio:%.4f' % (total_interactions, prop0/itemnum, prop1/itemnum))  
    #f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

    sampler = WarpSampler(user_train, usernum, itemnum, item_prop, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)

    model = Model(usernum, itemnum, args)    

    print(model)

    params = tf.trainable_variables()
    print('trainable vars:')
    for p in params:
        print(p)	

    T = 0.0
    t0 = time.time()
    
    u_final_val_ndcg = [0.0] * len(args.eval_k)
    u_final_val_hr = [0.0] * len(args.eval_k)
    u_final_test_ndcg = [0.0] * len(args.eval_k)
    u_final_test_hr = [0.0] * len(args.eval_k)
    u_best_epoch = 0

    temper = args.temper
    global_step = 0

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    with tf.Session(config=config) as sess:
        is_restore = init_model(sess, saver, args)
        if is_restore:
            t1 = time.time() - t0
            T += t1
            
            print ('Initial Evaluating')
            t_test = evaluate(model, dataset, args, sess, args.eval_k, False)
            t_valid = evaluate(model, dataset, args, sess, args.eval_k, True)
            print ('')
            print ('model results epoch:{}, time: {}(s), valid (NDCG@K: {}, HR@K: {}, U_NDCG@K: {}, U_HR@K: {}), test (NDCG@K: {}, HR@K: {}, U_NDCG@K: {}, U_HR@K: {})'.format(
                0, T, t_valid[0], t_valid[1], t_valid[2], t_valid[3], t_test[0], t_test[1], t_test[2], t_test[3]))
            u_best_epoch = 0
            u_final_val_ndcg = t_valid[2]
            u_final_val_hr = t_valid[3]
            u_final_test_ndcg = t_test[2]
            u_final_test_hr = t_test[3]

        try:
            for epoch in range(1, args.num_epochs + 1):
                print('start epoch: %d' % epoch)
                for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
                    if global_step >= num_batch * args.num_epochs:
                        print('global step equal to expected epoches')
                        break
                    
                    u, seq, pos, neg, pos_propensity, neg_propensity = sampler.next_batch()
                    global_step, auc, main_auc, int_match_auc, pop_auc, pop_match_auc, loss, main_loss, pop_loss, ortho_loss, int_match_loss, pop_match_loss, _ = \
                        sess.run([model.global_step, model.auc, model.main_auc, model.int_match_auc, model.pop_auc, model.pop_match_auc, model.loss, model.main_loss, model.pop_loss, model.ortho_loss, model.int_match_loss, model.pop_match_loss, model.train_op],
                                            {model.u: u, model.input_seq: seq, model.pos: pos, model.neg: neg, model.pos_propensity: pos_propensity, model.neg_propensity: neg_propensity,
                                            model.is_training: True})
                    if global_step % 100 == 0:    
                        print('global_step:{}, auc:{}, main_auc:{}, int_match_auc:{}, pop_auc:{}, pop_match_auc:{},  \
                            loss:{}, main_loss:{}, pop_loss:{}, \
                            ortho_loss:{}, int_match_loss:{}, pop_match_loss:{}'.format(\
                                global_step, auc, main_auc, int_match_auc, pop_auc, pop_match_auc, loss, main_loss, pop_loss, \
                                    ortho_loss, int_match_loss, pop_match_loss).strip())
                
                if epoch % args.eval_interval == 0 or global_step >= num_batch * args.num_epochs:
                    t1 = time.time() - t0
                    T += t1
                    
                    print ('Evaluating')
                    t_test = evaluate(model, dataset, args, sess, args.eval_k,False)
                    t_valid = evaluate(model, dataset, args, sess, args.eval_k, True)
                    print ('')
                    print ('model results epoch:{}, time: {}(s), valid (NDCG@K: {}, HR@K: {}, U_NDCG@K: {}, U_HR@K: {}), test (NDCG@K: {}, HR@K: {}, U_NDCG@K: {}, U_HR@K: {})'.format(
                        epoch, T, t_valid[0], t_valid[1], t_valid[2], t_valid[3], t_test[0], t_test[1], t_test[2], t_test[3]))
                    temper = temper - 1
                    if t_valid[2] > u_final_val_ndcg:
                        u_best_epoch = epoch
                        u_final_val_ndcg = t_valid[2]
                        u_final_val_hr = t_valid[3]
                        u_final_test_ndcg = t_test[2]
                        u_final_test_hr = t_test[3]
                        temper = args.temper
                        if args.model_dir.find('tmp_model') == -1:
                            checkpoint_path = os.path.join(args.model_dir, "model.ckpt")
                            saver.save(sess, checkpoint_path, global_step=model.global_step)    
                                
                    #f.write(str(t_valid) + ' ' + str(t_test) + '\n')
                    #f.flush()
                    t0 = time.time()
                if temper <= 0 or global_step >= num_batch * args.num_epochs:
                    print('early exit. temper:%d, global_step:%d' % (temper, global_step))
                    break
        except Exception as e:
            print(e)
            sampler.close()
            #f.close()
            exit(1)

        print("Done")
        model_result = 'unbiased best epoch:{}, valid((NDCG@K: {}, HR@K: {}), test (NDCG@K: {}, HR@K: {})'.format(
                    u_best_epoch, u_final_val_ndcg, u_final_val_hr, u_final_test_ndcg, u_final_test_hr)
        print(model_result)
        #f.close()
        sampler.close()


