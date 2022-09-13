from modules import *


class Model(object):
    def __init__(self, usernum, itemnum, args, reuse=None):
        self.is_training = tf.placeholder(tf.bool, shape=())
        self.u = tf.placeholder(tf.int32, shape=(None))
        self.input_seq = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.pos = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        self.neg = tf.placeholder(tf.int32, shape=(None, args.maxlen))
        pos = self.pos
        neg = self.neg
        print(args)
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, 0)), -1)        
        with tf.variable_scope(args.model, reuse=reuse):
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
            if args.disentangle:
                print('disentangel seq embedding')
                seq_interest, seq_pop_emb = self.disentangle_emb(seq, args)
                seq = seq_interest
            else:
                seq_pop_emb = seq
            
            if args.dynamic_seq_weight > 0:
                seq_pop_logits = self.predict_with_popularity_emb(seq_pop_emb, args)
                seq_pop_pred = tf.stop_gradient(tf.nn.sigmoid(seq_pop_logits))
                #seq_pop_pred = tf.nn.sigmoid(seq_pop_logits)
                
                if args.dynamic_seq_weight == 1:
                    seq_pop_pred = tf.maximum(seq_pop_pred, 0.02)
                    seq_w = 1 / seq_pop_pred
                else:
                    weight = self.weight_from_popularity_emb(seq_pop_emb, args)
                    seq_w = weight
                        
            self.seq = self.sequence_model(seq, mask, args, reuse, seq_w)
            
            if args.disentangle:
                self.seq_pop = self.sequence_model(seq_pop_emb, mask, args, reuse, 1/seq_w, 'pop_seq')
            else:
                self.seq_pop = self.seq
            

                
            pos = tf.reshape(pos, [tf.shape(self.input_seq)[0] * args.maxlen])
            neg = tf.reshape(neg, [tf.shape(self.input_seq)[0] * args.maxlen])
            pos_emb = tf.nn.embedding_lookup(item_emb_table, pos)
            neg_emb = tf.nn.embedding_lookup(item_emb_table, neg)
            
            if args.disentangle:
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
                # user embedding 
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
                    self.seq = user
                    self.seq_pop = user_pop
                else:
                    input = tf.concat([user, self.seq], axis=-1)
                    input = tf.reshape(input, [-1, args.maxlen, 2 * args.hidden_units])
                    u_seq_comb = feedforward(input, num_units=[args.hidden_units, args.hidden_units], 
                                scope='seq_user_combine_tower',
                                dropout_rate=args.dropout_rate, is_training=self.is_training,
                                use_residual=False,
                                reuse=tf.AUTO_REUSE)
                    u_seq_comb = tf.nn.relu(u_seq_comb)
                    self.seq += u_seq_comb

                    input = tf.concat([user_pop, self.seq_pop], axis=-1)
                    input = tf.reshape(input, [-1, args.maxlen, 2 * args.hidden_units])
                    u_seq_comb = feedforward(input, num_units=[args.hidden_units, args.hidden_units], 
                                scope='seq_user_pop_combine_tower',
                                dropout_rate=args.dropout_rate, is_training=self.is_training,
                                use_residual=False,
                                reuse=tf.AUTO_REUSE)
                    u_seq_comb = tf.nn.relu(u_seq_comb)
                    self.seq_pop += u_seq_comb
            if args.norm:
                self.seq = normalize(self.seq)
                if args.disentangle:
                    self.seq_pop = normalize(self.seq_pop, scope='ln-pop')
                else:
                    self.seq_pop = self.seq
                               
            self.seq = tf.reshape(self.seq, [tf.shape(self.input_seq)[0], args.maxlen, args.hidden_units])
            last_seq_emb =  self.seq[:, -1, :]
            print(last_seq_emb.get_shape())   
            last_seq_emb = tf.expand_dims(last_seq_emb, axis=1)  
            print(last_seq_emb.get_shape())   
            last_seq_pop_emb = self.seq_pop[:, -1, :]  
            last_seq_pop_emb = tf.expand_dims(last_seq_pop_emb, axis=1)              
            seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])
            seq_pop_emb = tf.reshape(self.seq_pop, [tf.shape(self.input_seq)[0] * args.maxlen, args.hidden_units])

            self.test_item = tf.placeholder(tf.int32, shape=(None, 101))
            test_item_emb = tf.nn.embedding_lookup(item_emb_table, self.test_item)
            
            if args.disentangle:
                test_item_emb, test_item_pop_emb = self.disentangle_emb(test_item_emb, args)
            else:
                test_item_pop_emb = test_item_emb
            
            #last_seq_emb = tf.reshape(last_seq_emb, [-1, args.hidden_units]) 
            #last_seq_pop_emb = tf.reshape(last_seq_pop_emb, [-1, args.hidden_units]) 
            self.test_logits = tf.matmul(last_seq_emb, tf.transpose(test_item_emb, perm=[0, 2, 1]))   
            
            if args.backbone > 0:
                #pred_emb = tf.expand_dims(last_seq_emb, axis=1)
                pred_emb = last_seq_emb
                pred_emb = tf.tile(pred_emb, [1, 101, 1])
                #pred_emb2 = tf.expand_dims(test_item_emb, axis=0)
                #pred_emb2 = tf.tile(pred_emb2, [tf.shape(self.input_seq)[0], 1, 1])
                pred_emb2 = test_item_emb
                pred_emb3 = pred_emb * pred_emb2
                pred_emb = tf.concat([pred_emb2, pred_emb, pred_emb3], axis=-1)
                pred_logits = self.predict_main_tower(pred_emb, args)
                pred_logits = tf.reshape(pred_logits, [tf.shape(self.input_seq)[0], 1, 101])
                self.test_logits += pred_logits
                
            if args.disentangle and args.pop_match_tower:
                pop_logits = tf.matmul(last_seq_pop_emb, tf.transpose(test_item_pop_emb, perm=[0, 2, 1]))
                if not args.dynamic_pop_int_weight:
                    self.test_logits += args.c1 * pop_logits
                else:
                    pred_emb = last_seq_emb
                    pred_emb = tf.tile(pred_emb, [1, 101, 1])
                    pred_pop_emb = last_seq_pop_emb
                    pred_pop_emb = tf.tile(pred_pop_emb, [1, 101, 1])
                    pred_emb = tf.concat([test_item_emb, test_item_pop_emb, pred_emb, pred_pop_emb], axis=-1)      
                    pop_match_weight = tf.nn.sigmoid(self.predict_main_tower(pred_emb, args, scope = 'predict_pop_match_weight')) 
                    pop_match_weight = tf.reshape(pop_match_weight, [tf.shape(self.input_seq)[0], 1, 101])          
                    self.test_logits = pop_match_weight * pop_logits + (1.0 - pop_match_weight) * self.test_logits
            self.test_logits = tf.reshape(self.test_logits, [tf.shape(self.input_seq)[0], 1, 101])

            if args.debias:
                test_item_pop_logits = self.predict_with_popularity_emb(test_item_pop_emb, args)
                test_item_pop_logits = tf.reshape(test_item_pop_logits, [-1, 1, 101])
                prob = tf.nn.sigmoid(self.test_logits) * tf.nn.sigmoid(test_item_pop_logits)
                last_seq_pop_logits = self.predict_with_popularity_emb(last_seq_pop_emb, args, scope='predict_with_seq_pop_emb')
                #last_seq_pop_logits = tf.expand_dims(last_seq_pop_logits, axis=-1)
                #self.test_logits = tf.log(prob / (1 - prob))
                if args.additive_bias:
                    self.test_logits += args.c0 * test_item_pop_logits
                else:
                    probBias = tf.nn.sigmoid(last_seq_pop_logits) * tf.nn.sigmoid(test_item_pop_logits)
                    #print(probBias.get_shape())
                    #print(self.test_logits.get_shape())
                    self.test_logits = (self.test_logits - args.c0) * probBias
            self.test_logits = self.test_logits[:, -1, :]

            # prediction layer
            self.pos_logits = tf.reduce_sum(pos_emb * seq_emb, -1)
            self.neg_logits = tf.reduce_sum(neg_emb * seq_emb, -1)
            self.pos_int_match_logits = self.pos_logits
            self.neg_int_match_logits = self.neg_logits
            
            if args.backbone > 0:
                pos_input = tf.concat([pos_emb, seq_emb, pos_emb * seq_emb], axis=-1)
                neg_input = tf.concat([neg_emb, seq_emb, neg_emb * seq_emb], axis=-1)
                
                pos_logits = tf.squeeze(self.predict_main_tower(pos_input, args), axis=-1)
                neg_logits = tf.squeeze(self.predict_main_tower(neg_input, args), axis=-1)
                self.pos_logits += pos_logits
                self.neg_logits += neg_logits
                self.pos_int_match_logits += pos_logits
                self.neg_int_match_logits += neg_logits
                
            if args.disentangle and args.pop_match_tower:
                pos_pop_match_logits = tf.reduce_sum(pos_pop_emb * seq_pop_emb, -1)
                neg_pop_match_logits = tf.reduce_sum(neg_pop_emb * seq_pop_emb, -1) 
                if not args.dynamic_pop_int_weight:
                    self.pos_logits += pos_pop_match_logits
                    self.neg_logits += neg_pop_match_logits
                else:
                    pos_input = tf.concat([pos_emb, pos_pop_emb, seq_emb, seq_pop_emb], axis=-1)
                    neg_input = tf.concat([neg_emb, neg_pop_emb, seq_emb, seq_pop_emb], axis=-1)
                    pos_pop_match_w = tf.squeeze(self.predict_main_tower(pos_input, args, scope = 'predict_pop_match_weight'), axis=-1)
                    pos_pop_match_w = tf.nn.sigmoid(pos_pop_match_w)
                    self.pos_logits = pos_pop_match_w * pos_pop_match_logits + (1 - pos_pop_match_w) * self.pos_logits
                    neg_pop_match_w = tf.squeeze(self.predict_main_tower(neg_input, args, scope = 'predict_pop_match_weight'), axis=-1)                       
                    neg_pop_match_w = tf.nn.sigmoid(neg_pop_match_w)
                    self.neg_logits = neg_pop_match_w * neg_pop_match_logits + (1 - neg_pop_match_w) * self.neg_logits
            else:
                pos_pop_match_logits = 0
                neg_pop_match_logits = 0            

                
            #pos_prob = tf.nn.sigmoid(self.pos_logits)
            #neg_prob = tf.nn.sigmoid(self.neg_logits)
            if args.debias:
                pos_pop_logits = self.predict_with_popularity_emb(pos_pop_emb, args)
                neg_pop_logits = self.predict_with_popularity_emb(neg_pop_emb, args)

                pos_int_logits = self.predict_with_popularity_emb(pos_emb, args, scope='predict_with_int_emb')
                neg_int_logits = self.predict_with_popularity_emb(neg_emb, args, scope='predict_with_int_emb')

                seq_pop_logits = self.predict_with_popularity_emb(seq_pop_emb, args, scope='predict_with_seq_pop_emb')
                
                pos_int_pop_match_emb = tf.concat([pos_pop_emb, seq_emb, pos_pop_emb * seq_emb], axis=-1)
                neg_int_pop_match_emb = tf.concat([neg_pop_emb, seq_emb, neg_pop_emb * seq_emb], axis=-1)
                
                pos_int_pop_match_logits = self.predict_main_tower(pos_int_pop_match_emb, args, scope='predict_int_pop_match')
                neg_int_pop_match_logits = self.predict_main_tower(neg_int_pop_match_emb, args, scope='predict_int_pop_match')
                                                
                pos_pop_logits = tf.reshape(pos_pop_logits, [tf.shape(self.input_seq)[0] * args.maxlen])
                neg_pop_logits = tf.reshape(neg_pop_logits, [tf.shape(self.input_seq)[0] * args.maxlen])
 
                pos_int_logits = tf.reshape(pos_int_logits, [tf.shape(self.input_seq)[0] * args.maxlen])
                neg_int_logits = tf.reshape(neg_int_logits, [tf.shape(self.input_seq)[0] * args.maxlen])
                seq_pop_logits = tf.reshape(seq_pop_logits, [tf.shape(self.input_seq)[0] * args.maxlen])
                pos_int_pop_match_logits = tf.reshape(pos_int_pop_match_logits, [tf.shape(self.input_seq)[0] * args.maxlen])
                neg_int_pop_match_logits = tf.reshape(neg_int_pop_match_logits, [tf.shape(self.input_seq)[0] * args.maxlen])
                                
                #pos_prob *= tf.nn.sigmoid(pos_pop_logits)
                #neg_prob *= tf.nn.sigmoid(neg_pop_logits)
                #print(self.pos_logits.get_shape())
                #print(pos_pop_logits.get_shape())
                #print(seq_pop_logits.get_shape())
                if args.additive_bias:
                    self.pos_logits += pos_pop_logits #+ seq_pop_logits
                    self.neg_logits += neg_pop_logits #+ seq_pop_logits
                else:
                    pos_prob = tf.nn.sigmoid(seq_pop_logits) * tf.nn.sigmoid(pos_pop_logits)
                    neg_prob = tf.nn.sigmoid(seq_pop_logits) * tf.nn.sigmoid(neg_pop_logits)
                    self.pos_logits *= pos_prob
                    self.neg_logits *= neg_prob
                #self.pos_logits += tf.stop_gradient(pos_pop_logits)
                #self.neg_logits += tf.stop_gradient(neg_pop_logits)
                
        # ignore padding items (0)
        istarget = tf.reshape(tf.to_float(tf.not_equal(pos, 0)), [tf.shape(self.input_seq)[0] * args.maxlen])
        if args.main_loss == 'point':
            self.loss = tf.reduce_sum(
                - tf.log(tf.sigmoid(self.pos_logits) + 1e-24) * istarget -
                tf.log(1 - tf.sigmoid(self.neg_logits) + 1e-24) * istarget
            ) / tf.reduce_sum(istarget)
        else:
            self.loss = tf.reduce_sum(
                - tf.log(tf.sigmoid(self.pos_logits - self.neg_logits) + 1e-24) * istarget 
            ) / tf.reduce_sum(istarget)
        self.main_loss = self.loss
        self.int_loss = tf.zeros([])
        self.int_match_loss = tf.zeros([])
        self.pop_match_loss = tf.zeros([])
        self.int_pop_match_loss = tf.zeros([])
        self.int_pop_match_kl_loss = tf.zeros([])
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
                if args.pop_match_tower:
                    pop_match_loss = tf.reduce_sum(
                        - tf.log(tf.sigmoid(pos_pop_match_logits) + 1e-24) * istarget -
                        tf.log(1 - tf.sigmoid(neg_pop_match_logits) + 1e-24) * istarget
                    ) / tf.reduce_sum(istarget)   
                    self.loss += args.pop_match_loss_w * pop_match_loss 
                    self.pop_match_loss = pop_match_loss               
                int_loss = tf.reduce_sum(
                    - tf.log(tf.sigmoid(pos_int_logits) + 1e-24) * istarget -
                    tf.log(1 - tf.sigmoid(neg_int_logits) + 1e-24) * istarget
                ) / tf.reduce_sum(istarget)    
                self.int_loss = int_loss
                if args.main_loss == 'point':
                    int_match_loss = tf.reduce_sum(
                        - tf.log(tf.sigmoid(self.pos_int_match_logits) + 1e-24) * istarget -
                        tf.log(1 - tf.sigmoid(self.neg_int_match_logits) + 1e-24) * istarget
                    ) / tf.reduce_sum(istarget) 
                else:
                    int_match_loss = tf.reduce_sum(
                        - tf.log(tf.sigmoid(self.pos_int_match_logits - self.neg_int_match_logits) + 1e-24) * istarget 
                    ) / tf.reduce_sum(istarget)                       
                self.int_match_loss = int_match_loss
                self.loss += args.int_match_loss_w * self.int_match_loss
        else:
            self.pop_loss = tf.zeros([])
            self.ortho_loss = tf.zeros([])
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_loss = sum(reg_losses)
        self.loss += self.reg_loss

        tf.summary.scalar('loss', self.loss)
        self.auc = tf.reduce_sum(
            ((tf.sign(self.pos_logits - self.neg_logits) + 1) / 2) * istarget
        ) / tf.reduce_sum(istarget)
        self.main_auc = tf.zeros([])
        self.main2_auc = tf.zeros([])
        self.pop_auc = tf.zeros([])
        self.pop_match_auc = tf.zeros([])
        self.int_auc = tf.zeros([])
        self.int_pop_match_auc = tf.zeros([])
        if args.debias:
            self.pop_auc = tf.reduce_sum(
                    ((tf.sign(pos_pop_logits - neg_pop_logits) + 1) / 2) * istarget
                ) / tf.reduce_sum(istarget)
            if args.additive_bias:
                self.main_auc = tf.reduce_sum(
                    ((tf.sign((self.pos_logits-pos_pop_logits) - (self.neg_logits-neg_pop_logits)) + 1) / 2) * istarget
                ) / tf.reduce_sum(istarget)
            else:
                self.main_auc = tf.reduce_sum(
                    ((tf.sign((self.pos_logits/pos_prob) - (self.neg_logits/neg_prob)) + 1) / 2) * istarget
                ) / tf.reduce_sum(istarget)                

            self.main2_auc = tf.reduce_sum(
                    ((tf.sign((self.pos_int_match_logits) - (self.neg_int_match_logits)) + 1) / 2) * istarget
                ) / tf.reduce_sum(istarget)
                                    
            if args.disentangle:
                if args.pop_match_tower:
                    self.pop_match_auc = tf.reduce_sum(
                            ((tf.sign(pos_pop_match_logits - neg_pop_match_logits) + 1) / 2) * istarget
                        ) / tf.reduce_sum(istarget)   
                    pos_int_pop_match_logits
                self.int_pop_match_auc = tf.reduce_sum(
                        ((tf.sign(pos_int_pop_match_logits - neg_int_pop_match_logits) + 1) / 2) * istarget
                    ) / tf.reduce_sum(istarget)
                self.int_auc = tf.reduce_sum(
                        ((tf.sign(pos_int_logits - neg_int_logits) + 1) / 2) * istarget
                    ) / tf.reduce_sum(istarget)                
        if reuse is None:
            tf.summary.scalar('auc', self.auc)
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=args.beta1, beta2=args.beta2)
            self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
            
            if args.debias and args.disentangle:
                updates, adv_updates = self.opt_interest_loss(args, self.optimizer)   
                updates2, adv_updates2 = self.opt_interest_pop_match_loss(args, pos_pop_logits, neg_pop_logits, 
                                    pos_int_pop_match_logits, neg_int_pop_match_logits, istarget, self.optimizer)   
                
                ops = [self.train_op, updates, updates2]            
                if args.adversarial and args.int_loss_w >0:
                    ops.append(adv_updates)
                if args.adversarial and args.int_pop_match_loss_w > 0:
                    ops.append(adv_updates2)
                
                self.train_op = tf.group(ops)
        else:
            tf.summary.scalar('test_auc', self.auc)

        self.merged = tf.summary.merge_all()

    def opt_interest_loss(self, args, opt=None):
        other_grad = []
        other_var = []
        adv_grad = []
        adv_var = []
        params = tf.trainable_variables()    
        gradients = tf.gradients(args.int_loss_w * self.int_loss, params) 
        
        if opt is None:
            opt = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=args.beta1, beta2=args.beta2)
            adv_opt = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=args.beta1, beta2=args.beta2)
        else:
            adv_opt = opt
        for grad,var in zip(gradients, params):
            if grad is not None:
                if var.op.name.find('predict_with_int_emb') != -1:
                    other_grad.append(grad)
                    other_var.append(var)
                elif var.op.name.find('interest_emb_disentangle') != -1:
                    adv_grad.append(-grad)
                    adv_var.append(var)  
        print('adversarial vars:')
        print(adv_var)
        print('other vars:')
        print(other_var)
        updates = opt.apply_gradients(zip(other_grad, other_var))
        adv_updates = adv_opt.apply_gradients(zip(adv_grad, adv_var))
        return updates, adv_updates

    def opt_interest_pop_match_loss(self, args, pos_pop_logits, neg_pop_logits, 
                                    pos_int_pop_match_logits, neg_int_pop_match_logits, istarget, opt=None):
        other_var = []
        adv_var = []
        params = tf.trainable_variables()    
        pos_int_pop_match_prob = tf.sigmoid(pos_int_pop_match_logits)
        neg_int_pop_match_prob = tf.sigmoid(neg_int_pop_match_logits)
        pos_pop_prob = tf.stop_gradient(tf.sigmoid(pos_pop_logits))
        neg_pop_prob = tf.stop_gradient(tf.sigmoid(neg_pop_logits))
        self.int_pop_match_loss = tf.reduce_sum(
            - tf.log(pos_int_pop_match_prob + 1e-24) * istarget -
            tf.log(1 - tf.sigmoid(neg_int_pop_match_prob) + 1e-24) * istarget
        ) / tf.reduce_sum(istarget)    
        
        if args.int_pop_match_loss_type == 0:
            kl_loss =  -pos_pop_prob * tf.log((pos_int_pop_match_prob+1e-24))   
            kl_loss +=  -(1 - pos_pop_prob) * tf.log(1.0 - pos_int_pop_match_prob + 1e-24)   
            kl_loss +=  -neg_pop_prob * tf.log((neg_int_pop_match_prob+1e-24))   
            kl_loss +=  -(1 - neg_pop_prob) * tf.log(1.0 - neg_int_pop_match_prob + 1e-24) 
        else:
            kl_loss =  pos_pop_prob * tf.log((pos_pop_prob+1e-24) / (pos_int_pop_match_prob+1e-24))   
            kl_loss +=  pos_int_pop_match_prob * tf.log((pos_int_pop_match_prob+1e-24) / (pos_pop_prob+1e-24))   
            kl_loss +=  neg_pop_prob * tf.log((neg_pop_prob+1e-24) / (neg_int_pop_match_prob+1e-24))   
            kl_loss +=  neg_int_pop_match_prob * tf.log((neg_int_pop_match_prob+1e-24) / (neg_pop_prob+1e-24))                
        
        self.int_pop_match_kl_loss = tf.reduce_sum(kl_loss * istarget) / tf.reduce_sum(istarget)
        if opt is None:
            opt = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=args.beta1, beta2=args.beta2)
            adv_opt = tf.train.AdamOptimizer(learning_rate=args.lr, beta1=args.beta1, beta2=args.beta2)
        else:
            adv_opt = opt
            
        for var in params:
            if var.op.name.find('predict_int_pop_match') != -1:
                other_var.append(var)
            elif var.op.name.find('pop_emb_disentangle') != -1:
                adv_var.append(var)  
        print('opt_interest_pop_match_loss adversarial vars:')
        print(adv_var)
        print('opt_interest_pop_match_loss other vars:')
        print(other_var)
        other_grad = tf.gradients(args.int_pop_match_loss_w * self.int_pop_match_loss, other_var)
        adv_grad = tf.gradients(args.int_pop_match_loss_w * self.int_pop_match_kl_loss, adv_var)
        other_grad, global_norm = tf.clip_by_global_norm(other_grad, clip_norm=2.0)
        adv_grad, global_norm = tf.clip_by_global_norm(adv_grad, clip_norm=2.0)
        #other_grad = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in other_grad]
        #adv_grad = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in adv_grad]
        updates = opt.apply_gradients(zip(other_grad, other_var))
        adv_updates = adv_opt.apply_gradients(zip(adv_grad, adv_var))
        #updates = opt.minimize(args.int_pop_match_loss_w * self.int_pop_match_loss, var_list=other_var)
        #adv_updates = adv_opt.minimize(args.int_pop_match_loss_w * self.int_pop_match_kl_loss, var_list=adv_var)
        return updates, adv_updates
            
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

    def weight_from_popularity_emb(self, pop, args):
        if pop.get_shape().ndims == 2:
            shape_change = True
            pop = tf.expand_dims(pop, axis=0)
        else:
            shape_change = False
        results = feedforward(pop, num_units=[2*args.hidden_units, 1], scope='weight_from_popularity_emb',
                                dropout_rate=args.dropout_rate, is_training=self.is_training,
                                use_residual=False,
                                reuse=tf.AUTO_REUSE)
        if shape_change:
            results = tf.squeeze(results, axis=0)
        return tf.sigmoid(results)
        
    def orthogonal_loss(self, emb0, emb1):
        dot = tf.abs(tf.reduce_sum(emb0 * emb1, -1))
        
        self_dot0 = tf.reduce_sum(emb0 * emb0, -1)
        self_dot1 = tf.reduce_sum(emb1 * emb1, -1)
        return tf.reduce_mean(dot / (tf.math.sqrt(self_dot0) * tf.math.sqrt(self_dot1) + 1e-24))
        
    def predict(self, sess, u, seq, item_idx):
        return sess.run(self.test_logits,
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