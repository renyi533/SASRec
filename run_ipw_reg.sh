i="0"
iter="5"
mkdir logs_ipw
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

    for config in ml-1m,200,0.2,200 Video,50,0.5,200 Steam,50,0.5,100 Electronics,50,0.5,100 
    do
        IFS=',' read dataset maxlen dropout epoch <<< "${config}"
        for model in SASRec GRU4Rec
        do
            python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout}   --model=${model} --num_epochs=${epoch} > logs_ipw/${dataset}_${model}_base_$i.log  2>&1 &

            python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=30 --disentangle=False --debias=True --additive_bias=False  > logs_ipw/${dataset}_${model}_macr_$i.log  2>&1 &

            python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=30 --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=False --pop_match_loss_w=0.0 --int_match_loss_w=0.0 --pop_match_tower=True > logs_ipw/${dataset}_${model}_dist_base_$i.log 2>&1 &

            python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=30 --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=False --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True > logs_ipw/${dataset}_${model}_dist_ipw_$i.log 2>&1 &

            wait_function

            python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=30 --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=True --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True > logs_ipw/${dataset}_${model}_dist_ipw_dyn_$i.log 2>&1 &

            python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=30 --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=False --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=False > logs_ipw/${dataset}_${model}_dist_ipw_nopop_$i.log 2>&1 &

            python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=30 --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=True --pop_match_loss_w=0.02 --int_match_loss_w=0.02  --pop_match_tower=True --ipw_loss=pair --main_loss=point --int_match_loss=pair > logs_ipw/${dataset}_${model}_dist_ipwpair_dyn_$i.log 2>&1 &

            wait_function
        done
    done
i=$[$i+1]
done

find logs_ipw -type f -name "*log" -exec awk '{s=$0};END{print FILENAME,s}' {} \; | sort