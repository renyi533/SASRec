i="0"
iter="5"
mkdir logs_ipw_beta_t
wait_function () {
    sleep 10
    for job in `jobs -p`
    do
        echo $job
        wait $job || let "FAIL+=1"
    done
}

while [ $i -lt $iter ]
do

    for config in ml-1m,200,0.2,200
    do
        IFS=',' read dataset maxlen dropout epoch <<< "${config}"
        for model in SASRec GRU4Rec
        do

        python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=30 --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.0002 --dynamic_pop_int_weight=True --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True > logs_ipw_beta_t/${dataset}_${model}_dist_ipw_0.0002_$i.log 2>&1 &

        python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=30 --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.002 --dynamic_pop_int_weight=True --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True > logs_ipw_beta_t/${dataset}_${model}_dist_ipw_0.002_$i.log 2>&1 &

        python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=30 --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=True --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True > logs_ipw_beta_t/${dataset}_${model}_dist_ipw_0.02_$i.log 2>&1 &

        python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=30 --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.2 --dynamic_pop_int_weight=True --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True > logs_ipw_beta_t/${dataset}_${model}_dist_ipw_0.2_$i.log 2>&1 &

        python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=30 --disentangle=True --debias=True --additive_bias=False --pop_loss_w=2.0 --dynamic_pop_int_weight=True --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True > logs_ipw_beta_t/${dataset}_${model}_dist_ipw_2.0_$i.log 2>&1 &


        wait_function

        done
    done
i=$[$i+1]
done

find logs_ipw_beta_t -type f -name "*log" -exec awk '{s=$0};END{print FILENAME,s}' {} \; | sort