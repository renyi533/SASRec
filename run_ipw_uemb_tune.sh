i="0"
iter="5"
mkdir logs_ipw_uemb_t
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

    for config in Electronics,50,0.5,100,70 ml-1m,200,0.2,200,30 Video,50,0.5,200,70
    do
        IFS=',' read dataset maxlen dropout epoch c0 <<< "${config}"
        for model in  SASRec  NextItRec  GRU4Rec
        do
        python main.py --enable_u=2 --backbone=0 --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=50 --maxlen=${maxlen} --dropout_rate=${dropout}   --model=${model} --num_epochs=${epoch}  > logs_ipw_uemb_t/${dataset}_${model}_base_withu_50_$i.log  2>&1 &
        python main.py --enable_u=2 --backbone=0 --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=25 --maxlen=${maxlen} --dropout_rate=${dropout}   --model=${model} --num_epochs=${epoch}  > logs_ipw_uemb_t/${dataset}_${model}_base_withu_25_$i.log  2>&1 &
        python main.py --enable_u=2 --backbone=0 --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=12 --maxlen=${maxlen} --dropout_rate=${dropout}   --model=${model} --num_epochs=${epoch}  > logs_ipw_uemb_t/${dataset}_${model}_base_withu_12_$i.log  2>&1 &
        python main.py --enable_u=2 --backbone=0 --dataset=${dataset} --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=${maxlen} --dropout_rate=${dropout}   --model=${model} --num_epochs=${epoch}  > logs_ipw_uemb_t/${dataset}_${model}_base_withu_6_$i.log  2>&1 &

        wait_function

        done
    done
i=$[$i+1]
done

find logs_ipw_uemb_t -type f -name "*log" -exec awk '{s=$0};END{print FILENAME,s}' {} \; | sort