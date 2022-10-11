i="0"
iter="3"
mkdir logs_ipw_c_t
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

    for config in Video,50,0.5,200 ml-1m,200,0.2,200
    do
        IFS=',' read dataset maxlen dropout epoch <<< "${config}"
        for model in SASRec GRU4Rec
        do

        python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=20 --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=True --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True > logs_ipw_c_t/${dataset}_${model}_dist_ipw_20_$i.log 2>&1 &

        python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=25 --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=True --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True > logs_ipw_c_t/${dataset}_${model}_dist_ipw_25_$i.log 2>&1 &

        python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=30 --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=True --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True > logs_ipw_c_t/${dataset}_${model}_dist_ipw_30_$i.log 2>&1 &

        python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=35 --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=True --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True > logs_ipw_c_t/${dataset}_${model}_dist_ipw_35_$i.log 2>&1 &

        python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=40 --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=True --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True > logs_ipw_c_t/${dataset}_${model}_dist_ipw_40_$i.log 2>&1 &


        wait_function

        done
    done
i=$[$i+1]
done

find logs_ipw_c_t -type f -name "*log" -exec awk '{s=$0};END{print FILENAME,s}' {} \; | sort