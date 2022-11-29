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
set -x
while [ $i -lt $iter ]
do

    for config in ml-1m,200,0.2,200,30,25,0.1 Video,50,0.5,200,70,25,0.15 Steam,50,0.5,100,30,6,0.05 Electronics,50,0.5,100,70,6,0.1 
    do
        IFS=',' read dataset maxlen dropout epoch c0 u_hidden_units pda_gamma <<< "${config}"
        for model in SASRec GRU4Rec NextItRec
        do
            python main.py --enable_u=0 --backbone=0 --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=${u_hidden_units} --maxlen=${maxlen} --dropout_rate=${dropout}   --model=${model} --num_epochs=${epoch} > logs_ipw/${dataset}_${model}_base_$i.log  2>&1 &

            python main.py --enable_u=0 --backbone=0 --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=${u_hidden_units} --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=${c0} --disentangle=False --debias=True --additive_bias=False  > logs_ipw/${dataset}_${model}_macr_$i.log  2>&1 &

            python main.py --enable_u=0 --backbone=0 --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=${u_hidden_units} --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=${c0} --disentangle=False --debias=True --additive_bias=False --pda_bias=True --main_loss=pair --ipw_factor=${pda_gamma} --ipw_min=0.01 > logs_ipw/${dataset}_${model}_pda_$i.log  2>&1 &

            python main.py --enable_u=0 --backbone=0 --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=${u_hidden_units} --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=${c0} --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=False --pop_match_loss_w=0.0 --int_match_loss_w=0.0 --pop_match_tower=True > logs_ipw/${dataset}_${model}_dist_base_$i.log 2>&1 &

            wait_function
            python main.py --enable_u=0 --backbone=0 --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=${u_hidden_units} --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=${c0} --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=False --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True > logs_ipw/${dataset}_${model}_dist_ipw_$i.log 2>&1 &

            python main.py --enable_u=0 --backbone=0 --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=${u_hidden_units} --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=0 --disentangle=False --debias=True --additive_bias=True > logs_ipw/${dataset}_${model}_biastower_$i.log  2>&1 &

            python main.py --enable_u=0 --backbone=0 --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=${u_hidden_units} --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=${c0} --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=True --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True > logs_ipw/${dataset}_${model}_dist_ipw_dyn_$i.log 2>&1 &
            wait_function

            python main.py --enable_u=0 --backbone=0 --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=${u_hidden_units} --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=${c0} --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=False --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=False > logs_ipw/${dataset}_${model}_dist_ipw_nopop_$i.log 2>&1 &

            python main.py --enable_u=0 --backbone=0 --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=${u_hidden_units} --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=${c0} --disentangle=False --debias=False --additive_bias=False --mode=ipw --main_loss=point > logs_ipw/${dataset}_${model}_ipw_point_$i.log 2>&1 &
            
            python main.py --enable_u=0 --backbone=0 --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=${u_hidden_units} --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=${c0} --disentangle=False --debias=False --additive_bias=False --mode=ipw --main_loss=pair > logs_ipw/${dataset}_${model}_ipw_pair_$i.log 2>&1 &
            wait_function

            python main.py --enable_u=0 --backbone=0 --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=${u_hidden_units} --maxlen=${maxlen} --dropout_rate=${dropout}   --model=${model} --num_epochs=${epoch} --main_loss=pair > logs_ipw/${dataset}_${model}_base_pair_$i.log  2>&1 &

            python main.py --enable_u=2 --backbone=0 --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=${u_hidden_units} --maxlen=${maxlen} --dropout_rate=${dropout}   --model=${model} --num_epochs=${epoch}  > logs_ipw/${dataset}_${model}_base_withu_$i.log  2>&1 &

            python main.py --enable_u=2 --backbone=0 --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=${u_hidden_units} --maxlen=${maxlen} --dropout_rate=${dropout} --model=${model} --num_epochs=${epoch} --c0=${c0} --disentangle=True --debias=True --additive_bias=False --pop_loss_w=0.02 --dynamic_pop_int_weight=True --pop_match_loss_w=0.02 --int_match_loss_w=0.02 --pop_match_tower=True > logs_ipw/${dataset}_${model}_dist_ipw_dyn_withu_$i.log 2>&1 &
            wait_function
        done
    done
i=$[$i+1]
done

find logs_ipw -type f -name "*log" -exec awk '{s=$0};END{print FILENAME,s}' {} \; | sort