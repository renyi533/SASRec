rm -rf logs*
mkdir logs
i="0"

wait_function () {
    sleep 10
    for job in `jobs -p`
    do
        echo $job
        wait $job || let "FAIL+=1"
    done
}

while [ $i -lt 2 ]
do


python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --model=GRU4Rec --num_epochs=350 > logs/ml_gru_rec_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --model=GRU4Rec --num_epochs=350 --enable_u=1 > logs/ml_uonly_rec_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --model=GRU4Rec --num_epochs=350 --enable_u=2 > logs/ml_gru_withu_rec_$i.log  2>&1 &

wait_function

python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2  --model=GRU4Rec --debias=True --num_epochs=350 --disentangle=False > logs/ml_gru_rec_debias_nodisentangle_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2  --model=GRU4Rec --debias=True --num_epochs=350 --disentangle=True --adversarial=True > logs/ml_gru_rec_debias_disentangle_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2  --model=GRU4Rec --debias=True --num_epochs=350 --disentangle=True --adversarial=False > logs/ml_gru_rec_debias_disentangle_noadv_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2  --model=GRU4Rec --debias=True --num_epochs=350 --disentangle=True --dynamic_seq_weight=1 > logs/ml_gru_rec_debias_disentangle_dynseq_$i.log  2>&1 &

wait_function

python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --model=SASRec --num_epochs=350 > logs/ml_sas_rec_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2 --model=SASRec --num_epochs=350 --enable_u=2 > logs/ml_sas_withu_rec_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2  --model=SASRec --debias=True --num_epochs=350 --disentangle=False > logs/ml_sas_rec_debias_nodisentangle_$i.log  2>&1 &
wait_function

python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2  --model=SASRec --debias=True --num_epochs=350 --disentangle=True --adversarial=True > logs/ml_sas_rec_debias_disentangle_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2  --model=SASRec --debias=True --num_epochs=350 --disentangle=True --adversarial=False > logs/ml_sas_rec_debias_disentangle_noadv_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --maxlen=200 --dropout_rate=0.2  --model=SASRec --debias=True --num_epochs=350 --disentangle=True --dynamic_seq_weight=1 > logs/ml_sas_rec_debias_disentangle_dynseq_$i.log  2>&1 &

wait_function
i=$[$i+1]
done

find logs -type f -name "*log" -exec awk '{s=$0};END{print FILENAME,s}' {} \; | sort
