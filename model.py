from modules import *


class Model(object):
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos_propensity = tf.placeholder(tf.float32, shape=(None, args.maxlen))
        self.neg_propensity = tf.placeholder(tf.float32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.test_item = tf.placeholder(tf.int32, shape=(None, 101))
        pos = self.pos
        neg = self.neg
        causal_scope = 'causal_scope'
        ipw_scope = 'ipw_scope'
        print(args)
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)       
        #reuse = tf.AUTO_REUSE 
        with tf.variable_scope(args.model, reuse=tf.AUTO_REUSE):
            seq, seq_pop, pos_emb, pos_pop_emb, neg_emb, neg_pop_emb, item_emb_table, pos_ortho_loss, neg_ortho_loss = \
                self.construct_seq_emb(args, itemnum, usernum, pos, neg, reuse, mask, args.disentangle, causal_scope)
            self.seq = seq
            self.seq_pop = seq_pop
            ipw_seq, _, ipw_pos_emb, _, ipw_neg_emb, _, ipw_item_emb_table, _, _ = \
                self.construct_seq_emb(args, itemnum, usernum, pos, neg, reuse, mask, False, ipw_scope)
            self.ipw_seq = ipw_seq
            
            self.test_logits = self.construct_test_logits(args, seq, seq_pop, item_emb_table, args.disentangle, args.debias, causal_scope, reuse)
            
            self.ipw_test_logits = self.construct_test_logits(args, ipw_seq, ipw_seq, ipw_item_emb_table, False, False, ipw_scope, reuse)
                                                 
            self.seq = tf.reshape(self.seq, [tf.shape(self.input_seq)[0], args.maxlen, args.hidden_units])
            
            seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])
            seq_pop_emb = tf.reshape(self.seq_pop, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])
            ipw_seq_emb = tf.reshape(self.ipw_seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])
            
            # prediction layer
            pos_logits, neg_logits, pos_int_match_logits, neg_int_match_logits, pos_pop_match_logits, neg_pop_match_logits, \
                pos_pop_logits, neg_pop_logits, seq_pop_logits = \
                    self.construct_train_logits(args, args.disentangle, args.debias, pos_emb, neg_emb, seq_emb, \
                                                pos_pop_emb, neg_pop_emb, seq_pop_emb, causal_scope, reuse)

            ipw_pos_logits, ipw_neg_logits, _, _, _, _, _, _, _ = \
                    self.construct_train_logits(args, False, False, ipw_pos_emb, ipw_neg_emb, ipw_seq_emb, \
                                                None, None, None, ipw_scope, reuse) 
                                   
        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_propensity = tf.reshape(self.pos_propensity, [tf.shape(self.input_seq)[0] * args.maxlen])
        neg_propensity = tf.reshape(self.neg_propensity, [tf.shape(self.input_seq)[0] * args.maxlen])
        pos_propensity = tf.maximum(tf.pow(pos_propensity, args.ipw_factor), args.ipw_min)
        neg_propensity = tf.maximum(tf.pow(1 - neg_propensity, args.ipw_factor), args.ipw_min)
        self.construct_causal_loss(args, pos_logits, neg_logits, istarget, pos_pop_logits, \
            neg_pop_logits, pos_ortho_loss, neg_ortho_loss, pos_int_match_logits, neg_int_match_logits,\
            pos_pop_match_logits, neg_pop_match_logits, pos_propensity, neg_propensity)
        self.construct_ipw_loss(args, ipw_pos_logits, ipw_neg_logits, istarget, pos_propensity, neg_propensity)
        self.construct_ipw_reg_loss(args, ipw_pos_logits, ipw_neg_logits, ipw_pos_emb, ipw_neg_emb, \
            pos_logits, neg_logits, pos_emb, neg_emb, istarget)
            
        tf.summary.scalar('loss', self.loss)
        self.auc = self.compute_auc(pos_logits, neg_logits, pos_propensity, istarget)
        self.ipw_auc = self.compute_auc(ipw_pos_logits, ipw_neg_logits, pos_propensity, istarget)
        self.main_auc = tf.zeros([])
        self.int_match_auc = tf.zeros([])
        self.pop_auc = tf.zeros([])
        self.pop_match_auc = tf.zeros([])
        if args.debias:
            self.pop_auc = self.compute_auc(pos_pop_logits, neg_pop_logits, pos_propensity, istarget)
            
            self.main_auc = self.compute_auc(pos_int_match_logits+pos_pop_match_logits, neg_int_match_logits+neg_pop_match_logits, pos_propensity, istarget)

            self.int_match_auc = self.compute_auc(pos_int_match_logits, neg_int_match_logits, pos_propensity, istarget)
                                    
            if args.disentangle:
                if args.pop_match_tower:
                    self.pop_match_auc = self.compute_auc(pos_pop_match_logits, neg_pop_match_logits, pos_propensity, istarget)   
        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=args.beta1, beta2=args.beta2)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()
    
    def construct_ipw_reg_loss(self, args, ipw_pos_logits, ipw_neg_logits, ipw_pos_emb, ipw_neg_emb, \
            pos_logits, neg_logits, pos_emb, neg_emb, istarget):
        ipw_pos_logits = tf.stop_gradient(ipw_pos_logits)
        ipw_neg_logits = tf.stop_gradient(ipw_neg_logits)
        ipw_pos_emb = tf.stop_gradient(ipw_pos_emb)
        ipw_neg_emb = tf.stop_gradient(ipw_neg_emb)

        ipw_pos_prob = tf.sigmoid(ipw_pos_logits)
        ipw_neg_prob = tf.sigmoid(ipw_neg_logits)
        pos_prob = tf.sigmoid(pos_logits)
        neg_prob = tf.sigmoid(neg_logits)
        
        ce_loss =  -ipw_pos_prob * tf.log((pos_prob+1e-24))   
        ce_loss +=  -(1 - ipw_pos_prob) * tf.log(1.0 - pos_prob + 1e-24)   
        ce_loss +=  -ipw_neg_prob * tf.log((neg_prob+1e-24))   
        ce_loss +=  -(1 - ipw_neg_prob) * tf.log(1.0 - neg_prob + 1e-24) 
        self.ipw_distillation_loss = tf.reduce_sum(ce_loss * istarget) / tf.reduce_sum(istarget)
        self.loss += args.ipw_distillation_loss_w * self.ipw_distillation_loss
        
        mse_loss = tf.reduce_sum(tf.math.square(ipw_pos_emb-pos_emb), axis=-1)
        mse_loss += tf.reduce_sum(tf.math.square(ipw_neg_emb-neg_emb), axis=-1)
        self.ipw_reg_loss = tf.reduce_sum(mse_loss * istarget) / tf.reduce_sum(istarget)
        self.loss += args.ipw_reg_loss_w * self.ipw_reg_loss
    
    def compute_auc(self, pos_logits, neg_logits, pos_propensity, istarget):
        auc = tf.reduce_sum(
                ((tf.sign((pos_logits) - (neg_logits)) + 1) / 2) * istarget
                        ) / tf.reduce_sum(istarget)
        
        u_auc = tf.reduce_sum(
                ((tf.sign((pos_logits) - (neg_logits)) + 1) / 2) * istarget / pos_propensity
                        ) / tf.reduce_sum(istarget / pos_propensity)
        return auc, u_auc
            
    def construct_ipw_loss(self, args, ipw_pos_logits, ipw_neg_logits, istarget, pos_propensity, neg_propensity):
        if args.ipw_loss == 'point':
            self.ipw_loss = tf.reduce_sum(
                - tf.log(tf.sigmoid(ipw_pos_logits) + 1e-24) / pos_propensity * istarget -
                tf.log(1 - tf.sigmoid(ipw_neg_logits) + 1e-24) / neg_propensity * istarget
            ) / tf.reduce_sum(istarget)
        else:
            self.ipw_loss = tf.reduce_sum(
                - tf.log(tf.sigmoid(ipw_pos_logits - ipw_neg_logits) + 1e-24) * istarget / pos_propensity
            ) / tf.reduce_sum(istarget)
        self.loss += self.ipw_loss
        
    def construct_causal_loss(self, args, pos_logits, neg_logits, istarget, pos_pop_logits, \
        neg_pop_logits, pos_ortho_loss, neg_ortho_loss, pos_int_match_logits, neg_int_match_logits,\
            pos_pop_match_logits, neg_pop_match_logits, pos_propensity, neg_propensity):
        if args.main_loss == 'point':
            self.loss = tf.reduce_sum(
                - tf.log(tf.sigmoid(pos_logits) + 1e-24) * istarget -
                tf.log(1 - tf.sigmoid(neg_logits) + 1e-24) * istarget
            ) / tf.reduce_sum(istarget)
        else:
            self.loss = tf.reduce_sum(
                - tf.log(tf.sigmoid(pos_logits - neg_logits) + 1e-24) * istarget 
            ) / tf.reduce_sum(istarget)
        self.main_loss = self.loss
        self.int_match_loss = tf.zeros([])
        self.pop_match_loss = tf.zeros([])
        self.pop_loss = tf.zeros([])
        self.ortho_loss = tf.zeros([])
        if args.debias:
            pop_loss = tf.reduce_sum(
                - tf.log(tf.sigmoid(pos_pop_logits) + 1e-24) * istarget -
                tf.log(1 - tf.sigmoid(neg_pop_logits) + 1e-24) * istarget
            ) / tf.reduce_sum(istarget)
            self.pop_loss = pop_loss
            self.ortho_loss = tf.zeros([]) + pos_ortho_loss + neg_ortho_loss
            self.loss += args.pop_loss_w * pop_loss
            self.loss += args.ortho_loss_w * (self.ortho_loss)
            if args.disentangle:
                if args.int_match_loss == 'point':
                    int_match_loss = tf.reduce_sum(
                        - tf.log(tf.sigmoid(pos_int_match_logits) + 1e-24) / pos_propensity * istarget -
                        tf.log(1 - tf.sigmoid(neg_int_match_logits) + 1e-24) / neg_propensity * istarget
                    ) / tf.reduce_sum(istarget) 
                else:
                    int_match_loss = tf.reduce_sum(
                        - tf.log(tf.sigmoid(pos_int_match_logits - neg_int_match_logits) + 1e-24) * istarget / pos_propensity
                    ) / tf.reduce_sum(istarget)                       
                self.int_match_loss = int_match_loss
                self.loss += args.int_match_loss_w * self.int_match_loss
                if args.pop_match_tower:
                    pop_match_loss = tf.reduce_sum(
                        - tf.log(tf.sigmoid(pos_pop_match_logits) + 1e-24) * istarget -
                        tf.log(1 - tf.sigmoid(neg_pop_match_logits) + 1e-24) * istarget
                    ) / tf.reduce_sum(istarget)   
                    self.loss += args.pop_match_loss_w * pop_match_loss 
                    self.pop_match_loss = pop_match_loss

        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_loss = sum(reg_losses)
        self.loss += self.reg_loss
        
    def construct_train_logits(self, args, disentangle, debias, pos_emb, neg_emb, 
                               seq_emb, pos_pop_emb, neg_pop_emb, seq_pop_emb, scope, reuse):
      with tf.variable_scope(scope, reuse=reuse): 
        pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
        neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)
        pos_int_match_logits = pos_logits
        neg_int_match_logits = neg_logits
        
        if args.backbone > 0:
            pos_input = tf.concat([pos_emb, seq_emb, pos_emb * seq_emb], axis=-1)
            neg_input = tf.concat([neg_emb, seq_emb, neg_emb * seq_emb], axis=-1)
            
            pos_logits_mlp = tf.squeeze(self.predict_main_tower(pos_input, args), axis=-1)
            neg_logits_mlp = tf.squeeze(self.predict_main_tower(neg_input, args), axis=-1)
            pos_logits += pos_logits_mlp
            neg_logits += neg_logits_mlp
            pos_int_match_logits += pos_logits_mlp
            neg_int_match_logits += neg_logits_mlp
            
        if disentangle and args.pop_match_tower:
            pos_pop_match_logits = tf.reduce_sum(pos_pop_emb * seq_pop_emb, -1)
            neg_pop_match_logits = tf.reduce_sum(neg_pop_emb * seq_pop_emb, -1) 
            if not args.dynamic_pop_int_weight:
                pos_logits += pos_pop_match_logits
                neg_logits += neg_pop_match_logits
            else:
                pos_input = tf.concat([pos_emb, pos_pop_emb, seq_emb, seq_pop_emb], axis=-1)
                neg_input = tf.concat([neg_emb, neg_pop_emb, seq_emb, seq_pop_emb], axis=-1)
                pos_pop_match_w = tf.squeeze(self.predict_main_tower(pos_input, args, scope = 'predict_pop_match_weight'), axis=-1)
                pos_pop_match_w = tf.nn.sigmoid(pos_pop_match_w)
                pos_logits = pos_pop_match_w * pos_pop_match_logits + (1 - pos_pop_match_w) * pos_logits
                neg_pop_match_w = tf.squeeze(self.predict_main_tower(neg_input, args, scope = 'predict_pop_match_weight'), axis=-1)                       
                neg_pop_match_w = tf.nn.sigmoid(neg_pop_match_w)
                neg_logits = neg_pop_match_w * neg_pop_match_logits + (1 - neg_pop_match_w) * neg_logits
        else:
            pos_pop_match_logits = 0
            neg_pop_match_logits = 0            
        
        pos_pop_logits = None
        neg_pop_logits = None
        seq_pop_logits = None
        if debias:
            pos_pop_logits = self.predict_with_popularity_emb(pos_pop_emb, args)
            neg_pop_logits = self.predict_with_popularity_emb(neg_pop_emb, args)
            seq_pop_logits = self.predict_with_popularity_emb(seq_pop_emb, args, scope='predict_with_seq_pop_emb')
            
            pos_pop_logits = tf.reshape(pos_pop_logits, [tf.shape(self.input_seq)[0] * args.maxlen])
            neg_pop_logits = tf.reshape(neg_pop_logits, [tf.shape(self.input_seq)[0] * args.maxlen])
            seq_pop_logits = tf.reshape(seq_pop_logits, [tf.shape(self.input_seq)[0] * args.maxlen])
                            
            if args.additive_bias:
                pos_logits += pos_pop_logits #+ seq_pop_logits
                neg_logits += neg_pop_logits #+ seq_pop_logits
            else:
                pos_prob = tf.nn.sigmoid(seq_pop_logits) * tf.nn.sigmoid(pos_pop_logits)
                neg_prob = tf.nn.sigmoid(seq_pop_logits) * tf.nn.sigmoid(neg_pop_logits)
                pos_logits *= pos_prob
                neg_logits *= neg_prob 

        return pos_logits, neg_logits, pos_int_match_logits, neg_int_match_logits, pos_pop_match_logits, neg_pop_match_logits,\
            pos_pop_logits, neg_pop_logits, seq_pop_logits
        
    def construct_seq_emb(self, args, itemnum, usernum, pos, neg, reuse, mask, disentangle, scope):
        with tf.variable_scope(scope, reuse=reuse):
            # sequence embedding, item embedding table
            seq, item_emb_table = embedding(self.input_seq,
                                            vocab_size=itemnum + 1,
                                            num_units=args.hidden_units,
                                            zero_pad=True,
                                            scale=True,
                                            l2_reg=args.l2_emb,
                                            scope="input_embeddings",
                                            with_t=True,
                                            reuse=reuse
                                            )
            seq_w = tf.ones([tf.shape(seq)[0], tf.shape(seq)[1], 1])
            if disentangle:
                print('disentangel seq embedding')
                seq_interest, seq_pop_emb = self.disentangle_emb(seq, args)
                seq = seq_interest
            else:
                seq_pop_emb = seq
            
            seq = self.sequence_model(seq, mask, args, reuse, seq_w)
            
            if disentangle:
                seq_pop = self.sequence_model(seq_pop_emb, mask, args, reuse, seq_w, 'pop_seq')
            else:
                seq_pop = seq
                
            pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
            neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
            pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
            neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
            
            if disentangle:
                print('disentangle pos/neg embedding')
                pos_emb, pos_pop_emb = self.disentangle_emb(pos_emb, args)
                neg_emb, neg_pop_emb = self.disentangle_emb(neg_emb, args)
                pos_ortho_loss = self.orthogonal_loss(pos_emb, pos_pop_emb)
                neg_ortho_loss = self.orthogonal_loss(neg_emb, neg_pop_emb)
            else:
                pos_pop_emb = pos_emb
                neg_pop_emb = neg_emb
                pos_ortho_loss = 0.0
                neg_ortho_loss = 0.0
            
            if args.enable_u > 0:
                seq, seq_pop = self.merge_user_seq_emb(args, seq, seq_pop, usernum, reuse)
                
            if args.norm:
                seq = normalize(seq)
                if disentangle:
                    seq_pop = normalize(seq_pop, scope='ln-pop')
                else:
                    seq_pop = seq
            return seq, seq_pop, pos_emb, pos_pop_emb, neg_emb, neg_pop_emb, item_emb_table, pos_ortho_loss, neg_ortho_loss
               
    def construct_test_logits(self, args, seq, seq_pop, item_emb_table, disentangle, debias, scope, reuse):
        with tf.variable_scope(scope, reuse=reuse):
            
            last_seq_emb =  seq[:, -1, :]
            last_seq_emb = tf.expand_dims(last_seq_emb, axis=1)  
            last_seq_pop_emb = seq_pop[:, -1, :]  
            last_seq_pop_emb = tf.expand_dims(last_seq_pop_emb, axis=1)              
            
            test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
            
            if disentangle:
                test_item_emb, test_item_pop_emb = self.disentangle_emb(test_item_emb, args)
            else:
                test_item_pop_emb = test_item_emb
            
            test_logits = tf.matmul(last_seq_emb, tf.transpose(test_item_emb, perm=[0, 2, 1]))   
            
            if args.backbone > 0:
                pred_emb = last_seq_emb
                pred_emb = tf.tile(pred_emb, [1, 101, 1])
                pred_emb2 = test_item_emb
                pred_emb3 = pred_emb * pred_emb2
                pred_emb = tf.concat([pred_emb2, pred_emb, pred_emb3], axis=-1)
                pred_logits = self.predict_main_tower(pred_emb, args)
                pred_logits = tf.reshape(pred_logits, [tf.shape(self.input_seq)[0], 1, 101])
                test_logits += pred_logits
                
            if disentangle and args.pop_match_tower:
                pop_logits = tf.matmul(last_seq_pop_emb, tf.transpose(test_item_pop_emb, perm=[0, 2, 1]))
                if not args.dynamic_pop_int_weight:
                    test_logits += pop_logits
                else:
                    pred_emb = last_seq_emb
                    pred_emb = tf.tile(pred_emb, [1, 101, 1])
                    pred_pop_emb = last_seq_pop_emb
                    pred_pop_emb = tf.tile(pred_pop_emb, [1, 101, 1])
                    pred_emb = tf.concat([test_item_emb, test_item_pop_emb, pred_emb, pred_pop_emb], axis=-1)      
                    pop_match_weight = tf.nn.sigmoid(self.predict_main_tower(pred_emb, args, scope = 'predict_pop_match_weight')) 
                    pop_match_weight = tf.reshape(pop_match_weight, [tf.shape(self.input_seq)[0], 1, 101])          
                    test_logits = pop_match_weight * pop_logits + (1.0 - pop_match_weight) * test_logits
            test_logits = tf.reshape(test_logits, [tf.shape(self.input_seq)[0], 1, 101])

            if debias:
                test_item_pop_logits = self.predict_with_popularity_emb(test_item_pop_emb, args)
                test_item_pop_logits = tf.reshape(test_item_pop_logits, [-1, 1, 101])
                prob = tf.nn.sigmoid(test_logits) * tf.nn.sigmoid(test_item_pop_logits)
                last_seq_pop_logits = self.predict_with_popularity_emb(last_seq_pop_emb, args, scope='predict_with_seq_pop_emb')
                if args.additive_bias:
                    test_logits += args.c0 * test_item_pop_logits
                else:
                    probBias = tf.nn.sigmoid(last_seq_pop_logits) * tf.nn.sigmoid(test_item_pop_logits)
                    test_logits = (test_logits - args.c0) * probBias
            test_logits = test_logits[:, -1, :]
            return test_logits
        
    def merge_user_seq_emb(self, args, seq, seq_pop, usernum, reuse):
        user, user_emb_table = embedding(self.u,
                                        vocab_size=usernum + 1,
                                        num_units=args.u_hidden_units,
                                        zero_pad=True,
                                        scale=True,
                                        l2_reg=args.l2_emb,
                                        scope="user_embeddings",
                                        with_t=True,
                                        reuse=reuse
                                        )
        user = tf.expand_dims(user, axis=1)
        user = tf.tile(user, [1, args.maxlen,1])
        user = tf.reshape(user, [-1, args.maxlen, args.u_hidden_units])
        
        if args.disentangle:
            user, user_pop = self.disentangle_emb(user, args, scope='disentangle_user_emb')
        else:
            if args.u_hidden_units != args.hidden_units:
                user = tf.layers.dense(user, units=args.hidden_units)
            user_pop = user
            
        if args.enable_u == 1:
            seq = user
            seq_pop = user_pop
        else:
            input = tf.concat([user, seq], axis=-1)
            input = tf.reshape(input, [-1, args.maxlen, 2 * args.hidden_units])
            u_seq_comb = feedforward(input, num_units=[args.hidden_units, args.hidden_units], 
                        scope='seq_user_combine_tower',
                        dropout_rate=args.dropout_rate, is_training=self.is_training,
                        use_residual=False,
                        reuse=tf.AUTO_REUSE)
            u_seq_comb = tf.nn.relu(u_seq_comb)
            seq += u_seq_comb

            input = tf.concat([user_pop, seq_pop], axis=-1)
            input = tf.reshape(input, [-1, args.maxlen, 2 * args.hidden_units])
            u_seq_comb = feedforward(input, num_units=[args.hidden_units, args.hidden_units], 
                        scope='seq_user_pop_combine_tower',
                        dropout_rate=args.dropout_rate, is_training=self.is_training,
                        use_residual=False,
                        reuse=tf.AUTO_REUSE)
            u_seq_comb = tf.nn.relu(u_seq_comb)
            seq_pop += u_seq_comb
        return seq, seq_pop
                 
    def sequence_model(self, seq_in, mask, args, reuse, seq_w, scope='interest_seq'):
        with tf.variable_scope(scope, reuse=reuse):
            if args.model == 'SASRec':
                print('SASRec')
                seq = self.sas_seq_gen(seq_in, mask, args, reuse, seq_w)
            elif args.model == 'GRU4Rec':
                print('GRU4Rec')
                seq = self.gru_seq_gen(seq_in, mask, args, reuse, seq_w)
            elif args.model == 'SumPoolingRec':
                print('SumPoolingRec')
                seq = self.sumpooling_seq_gen(seq_in, mask, args, reuse, seq_w)
            else:
                print('Warning! use original seq to predict')
                seq = seq_in 
            return seq       
        
    def disentangle_emb(self, seq, args, scope='disentangle_emb'):
        if seq.get_shape().ndims == 2:
            shape_change = True
            seq = tf.expand_dims(seq, axis=0)
        else:
            shape_change = False
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):    
            interest = feedforward(seq, num_units=[3*args.hidden_units, args.hidden_units], 
                                    scope='interest_emb_disentangle',
                                    dropout_rate=0, is_training=self.is_training,
                                    use_residual=False,
                                    reuse=tf.AUTO_REUSE)
            
            popularity = feedforward(seq, num_units=[2*args.hidden_units, args.hidden_units], 
                                    scope='pop_emb_disentangle',
                                    dropout_rate=0, is_training=self.is_training,
                                    use_residual=False,
                                    reuse=tf.AUTO_REUSE)
        #interest = tf.nn.relu(interest)
        #popularity = tf.nn.relu(popularity)
        if shape_change:
            popularity = tf.squeeze(popularity, axis=0)
            interest = tf.squeeze(interest, axis=0)
        return interest, popularity

    def predict_with_popularity_emb(self, pop, args, scope='predict_with_popularity_emb'):
        if pop.get_shape().ndims == 2:
            shape_change = True
            pop = tf.expand_dims(pop, axis=0)
        else:
            shape_change = False
        results = feedforward(pop, num_units=[2*args.hidden_units, 1], scope=scope,
                                dropout_rate=args.dropout_rate, is_training=self.is_training,
                                use_residual=False,
                                reuse=tf.AUTO_REUSE)
        if shape_change:
            results = tf.squeeze(results, axis=0)
        return results
    
    def predict_main_tower(self, input, args, scope = 'predict_main_tower'):
        if input.get_shape().ndims == 2:
            shape_change = True
            input = tf.expand_dims(input, axis=0)
        else:
            shape_change = False
        results = feedforward(input, num_units=[2*args.hidden_units, 1], scope=scope,
                                dropout_rate=args.dropout_rate, is_training=self.is_training,
                                use_residual=False,
                                reuse=tf.AUTO_REUSE)
        if shape_change:
            results = tf.squeeze(results, axis=0)
        return results

    def orthogonal_loss(self, emb0, emb1):
        dot = tf.abs(tf.reduce_sum(emb0 * emb1, -1))
        
        self_dot0 = tf.reduce_sum(emb0 * emb0, -1)
        self_dot1 = tf.reduce_sum(emb1 * emb1, -1)
        return tf.reduce_mean(dot / (tf.math.sqrt(self_dot0) * tf.math.sqrt(self_dot1) + 1e-24))
        
    def predict(self, sess, u, seq, item_idx):
        return sess.run([self.test_logits, self.ipw_test_logits],
                        {self.u: u, self.input_seq: seq, self.test_item: item_idx, self.is_training: False})

    def sas_seq_gen(self, seq, mask, args, reuse, seq_w):
        # sequence embedding, item embedding table
        seq *= seq_w
        # Positional Encoding
        t, pos_emb_table = embedding(
            tf.tile(tf.expand_dims(tf.range(tf.shape(self.input_seq)[1]), 0), [tf.shape(self.input_seq)[0], 1]),
            vocab_size=args.maxlen,
            num_units=args.hidden_units,
            zero_pad=False,
            scale=False,
            l2_reg=args.l2_emb,
            scope="dec_pos",
            reuse=reuse,
            with_t=True
        )
        seq += t

        # Dropout
        seq = tf.layers.dropout(seq,
                                rate=args.dropout_rate,
                                training=tf.convert_to_tensor(self.is_training))
        seq *= mask

        # Build blocks

        for i in range(args.num_blocks):
            with tf.variable_scope("num_blocks_%d" % i):

                # Self-attention
                seq = multihead_attention(queries=normalize(seq),
                                          keys=seq,
                                          num_units=args.hidden_units,
                                          num_heads=args.num_heads,
                                          dropout_rate=args.dropout_rate,
                                          is_training=self.is_training,
                                          causality=True,
                                          scope="self_attention",
                                          keys_w=None)
                # Feed forward
                seq = feedforward(normalize(seq), num_units=[args.hidden_units, args.hidden_units],
                                       dropout_rate=args.dropout_rate, is_training=self.is_training)
                seq *= mask
        return seq

    def gru_seq_gen(self, seq, mask, args, reuse, seq_w):
        # Dropout
        seq *= seq_w
        seq = tf.layers.dropout(seq,
                                rate=args.dropout_rate,
                                training=tf.convert_to_tensor(self.is_training))
        seq *= mask

        # Build blocks
        layers = [args.hidden_units] * args.num_blocks
        rnn_layers = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(size), output_keep_prob=1.0) for size in layers]
        #rnn_layers = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(size), output_keep_prob=1.0-args.dropout_rate) for size in layers]
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
        outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                           inputs=seq,
                                           dtype=tf.float32)            
        outputs *= mask
        return outputs

    def sumpooling_seq_gen(self, seq, mask, args, reuse, seq_w):
        seq *= seq_w
        # Dropout
        seq = tf.layers.dropout(seq,
                                rate=args.dropout_rate,
                                training=tf.convert_to_tensor(self.is_training))
        seq *= mask

        # Build blocks
        outputs = tf.math.cumsum(seq, axis=1)
                  
        outputs *= mask
        return outputs