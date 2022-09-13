rm -rf logs_debias*
mkdir logs_debias
i="0"
iter="5"
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

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=GRU4Rec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=1.0 --pop_match_tower=False > logs_debias/ml-1m_gru_rec_dist_nopopmatch_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=SASRec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=1.0 --pop_match_tower=False > logs_debias/ml-1m_sas_rec_dist_nopopmatch_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2   --model=GRU4Rec --num_epochs=200 > logs_debias/ml-1m_gru_rec_base_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2   --model=SASRec --num_epochs=200 > logs_debias/ml-1m_sas_rec_base_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=GRU4Rec --num_epochs=200 --c0=0 --disentangle=False --debias=True --adversarial=False --additive_bias=True  > logs_debias/ml-1m_gru_rec_debias_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=SASRec --num_epochs=200 --c0=0 --disentangle=False --debias=True --adversarial=False --additive_bias=True  > logs_debias/ml-1m_sas_rec_debias_$i.log  2>&1 &

wait_function

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=GRU4Rec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=1.0 > logs_debias/ml-1m_gru_rec_dist_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=SASRec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=1.0 > logs_debias/ml-1m_sas_rec_dist_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=GRU4Rec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=0.8 > logs_debias/ml-1m_gru_rec_dist_r0.8_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=SASRec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=0.8 > logs_debias/ml-1m_sas_rec_dist_r0.8_$i.log  2>&1 &

#python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=GRU4Rec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=True --additive_bias=True --c1=0.8 --int_pop_match_loss_w=0.5 --int_loss_w=0 > logs_debias/ml-1m_gru_rec_dist_r0.8_popadv_$i.log  2>&1 &

#python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=SASRec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=True --additive_bias=True --c1=0.8 --int_pop_match_loss_w=0.5 --int_loss_w=0 > logs_debias/ml-1m_sas_rec_dist_r0.8_popadv_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2   --model=SASRec --num_epochs=200 --main_loss=pair > logs_debias/ml-1m_sas_rec_base_pair_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=SASRec --num_epochs=200 --c0=0 --disentangle=False --debias=True --adversarial=False --additive_bias=True --main_loss=pair  > logs_debias/ml-1m_sas_rec_debias_pair_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=SASRec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=1.0 --pop_match_tower=False --main_loss=pair > logs_debias/ml-1m_sas_rec_dist_nopopmatch_pair_$i.log  2>&1 &

wait_function


python main.py --dataset=Beauty --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=1.0 --pop_match_tower=False > logs_debias/beauty_gru_rec_dist_nopopmatch_$i.log  2>&1 &

python main.py --dataset=Beauty --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=1.0 --pop_match_tower=False > logs_debias/beauty_sas_rec_dist_nopopmatch_$i.log  2>&1 &

python main.py --dataset=Beauty --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5   --model=GRU4Rec --num_epochs=200 > logs_debias/beauty_gru_rec_base_$i.log  2>&1 &

python main.py --dataset=Beauty --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5   --model=SASRec --num_epochs=200 > logs_debias/beauty_sas_rec_base_$i.log  2>&1 &

python main.py --dataset=Beauty --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=0 --disentangle=False --debias=True --adversarial=False --additive_bias=True  > logs_debias/beauty_gru_rec_debias_$i.log  2>&1 &

python main.py --dataset=Beauty --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=0 --disentangle=False --debias=True --adversarial=False --additive_bias=True  > logs_debias/beauty_sas_rec_debias_$i.log  2>&1 &

wait_function

python main.py --dataset=Beauty --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=1.0 > logs_debias/beauty_gru_rec_dist_$i.log  2>&1 &

python main.py --dataset=Beauty --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=1.0 > logs_debias/beauty_sas_rec_dist_$i.log  2>&1 &

python main.py --dataset=Beauty --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=0.8 > logs_debias/beauty_gru_rec_dist_r0.8_$i.log  2>&1 &

python main.py --dataset=Beauty --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=0.8 > logs_debias/beauty_sas_rec_dist_r0.8_$i.log  2>&1 &

#python main.py --dataset=Beauty --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=True --additive_bias=True --c1=0.8 --int_pop_match_loss_w=0.5 --int_loss_w=0 > logs_debias/beauty_gru_rec_dist_r0.8_popadv_$i.log  2>&1 &

#python main.py --dataset=Beauty --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=True --additive_bias=True --c1=0.8 --int_pop_match_loss_w=0.5 --int_loss_w=0 > logs_debias/beauty_sas_rec_dist_r0.8_popadv_$i.log  2>&1 &

python main.py --dataset=Beauty --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5   --model=SASRec --num_epochs=200 --main_loss=pair > logs_debias/Beauty_sas_rec_base_pair_$i.log  2>&1 &

python main.py --dataset=Beauty --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=0 --disentangle=False --debias=True --adversarial=False --additive_bias=True --main_loss=pair  > logs_debias/Beauty_sas_rec_debias_pair_$i.log  2>&1 &

python main.py --dataset=Beauty --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=1.0 --pop_match_tower=False --main_loss=pair > logs_debias/Beauty_sas_rec_dist_nopopmatch_pair_$i.log  2>&1 &
wait_function


python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=1.0 --pop_match_tower=False > logs_debias/steam_gru_rec_dist_nopopmatch_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=1.0 --pop_match_tower=False > logs_debias/steam_sas_rec_dist_nopopmatch_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5   --model=GRU4Rec --num_epochs=200 > logs_debias/steam_gru_rec_base_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5   --model=SASRec --num_epochs=200 > logs_debias/steam_sas_rec_base_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=0 --disentangle=False --debias=True --adversarial=False --additive_bias=True  > logs_debias/steam_gru_rec_debias_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=0 --disentangle=False --debias=True --adversarial=False --additive_bias=True  > logs_debias/steam_sas_rec_debias_$i.log  2>&1 &

wait_function

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=1.0 > logs_debias/steam_gru_rec_dist_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=1.0 > logs_debias/steam_sas_rec_dist_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=0.8 > logs_debias/steam_gru_rec_dist_r0.8_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=0.8 > logs_debias/steam_sas_rec_dist_r0.8_$i.log  2>&1 &

#python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=True --additive_bias=True --c1=0.8 --int_pop_match_loss_w=0.5 --int_loss_w=0 > logs_debias/steam_gru_rec_dist_r0.8_popadv_$i.log  2>&1 &

#python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=True --additive_bias=True --c1=0.8 --int_pop_match_loss_w=0.5 --int_loss_w=0 > logs_debias/steam_sas_rec_dist_r0.8_popadv_$i.log  2>&1 &
python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5   --model=SASRec --num_epochs=200 --main_loss=pair > logs_debias/Steam_sas_rec_base_pair_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=0 --disentangle=False --debias=True --adversarial=False --additive_bias=True --main_loss=pair  > logs_debias/Steam_sas_rec_debias_pair_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=1.0 --pop_match_tower=False --main_loss=pair > logs_debias/Steam_sas_rec_dist_nopopmatch_pair_$i.log  2>&1 &

wait_function


python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=1.0 --pop_match_tower=False > logs_debias/video_gru_rec_dist_nopopmatch_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=1.0 --pop_match_tower=False > logs_debias/video_sas_rec_dist_nopopmatch_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5   --model=GRU4Rec --num_epochs=200 > logs_debias/video_gru_rec_base_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5   --model=SASRec --num_epochs=200 > logs_debias/video_sas_rec_base_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=0 --disentangle=False --debias=True --adversarial=False --additive_bias=True  > logs_debias/video_gru_rec_debias_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=0 --disentangle=False --debias=True --adversarial=False --additive_bias=True  > logs_debias/video_sas_rec_debias_$i.log  2>&1 &

wait_function

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=1.0 > logs_debias/video_gru_rec_dist_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=1.0 > logs_debias/video_sas_rec_dist_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=0.8 > logs_debias/video_gru_rec_dist_r0.8_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=0.8 > logs_debias/video_sas_rec_dist_r0.8_$i.log  2>&1 &

#python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=True --additive_bias=True --c1=0.8 --int_pop_match_loss_w=0.5 --int_loss_w=0 > logs_debias/video_gru_rec_dist_r0.8_popadv_$i.log  2>&1 &

#python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=True --additive_bias=True --c1=0.8 --int_pop_match_loss_w=0.5 --int_loss_w=0 > logs_debias/video_sas_rec_dist_r0.8_popadv_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5   --model=SASRec --num_epochs=200 --main_loss=pair > logs_debias/Video_sas_rec_base_pair_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=0 --disentangle=False --debias=True --adversarial=False --additive_bias=True --main_loss=pair  > logs_debias/Video_sas_rec_debias_pair_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=0 --disentangle=True --debias=True --adversarial=False --additive_bias=True --c1=1.0 --pop_match_tower=False --main_loss=pair > logs_debias/Video_sas_rec_dist_nopopmatch_pair_$i.log  2>&1 &

wait_function

i=$[$i+1]
done

find logs_debias -type f -name "*log" -exec awk '{s=$0};END{print FILENAME,s}' {} \; | sort