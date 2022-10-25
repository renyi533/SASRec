i="0"
iter="5"
mkdir logs_ipw_ortho_t
wait_function () {
    sleep 10
    for job in `jobs -p`
    do
        echo $job
        wait $job || let "FAIL+=1"
    done
}
set -x
while [ $i -lt $iter ]
do

    for config in ml-1m,200,0.2,200,30 Video,50,0.5,200,70
    do
        IFS=',' read dataset maxlen dropout epoch c0 <<< "${config}"
        for model in SASRec GRU4Rec
        do
        python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=${c0} --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=True --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True --ortho_loss_w=0.0 > logs_ipw_ortho_t/${dataset}_${model}_dist_ipw_0.0_$i.log 2>&1 &

        python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=${c0} --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=True --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True --ortho_loss_w=0.0005 > logs_ipw_ortho_t/${dataset}_${model}_dist_ipw_0.0005_$i.log 2>&1 &

        python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=${c0} --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=True --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True --ortho_loss_w=0.005 > logs_ipw_ortho_t/${dataset}_${model}_dist_ipw_0.005_$i.log 2>&1 &

        python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=${c0} --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=True --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True --ortho_loss_w=0.05 > logs_ipw_ortho_t/${dataset}_${model}_dist_ipw_0.05_$i.log 2>&1 &
        wait_function

        python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=${c0} --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=True --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True --ortho_loss_w=0.5 > logs_ipw_ortho_t/${dataset}_${model}_dist_ipw_0.5_$i.log 2>&1 &

        python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=${c0} --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=True --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True --ortho_loss_w=5 > logs_ipw_ortho_t/${dataset}_${model}_dist_ipw_5_$i.log 2>&1 &

        python main.py --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=${c0} --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=True --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True --ortho_loss_w=50 > logs_ipw_ortho_t/${dataset}_${model}_dist_ipw_50_$i.log 2>&1 &


        wait_function

        done
    done
i=$[$i+1]
done

find logs_ipw_ortho_t -type f -name "*log" -exec awk '{s=$0};END{print FILENAME,s}' {} \; | sort