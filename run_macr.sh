rm -rf logs_macr*
mkdir logs_macr
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

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=1.0 --pop_match_tower=False > logs_macr/ml-1m_gru_rec_dist_nopopmatch_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=1.0 --pop_match_tower=False > logs_macr/ml-1m_sas_rec_dist_nopopmatch_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2   --model=GRU4Rec --num_epochs=200 > logs_macr/ml-1m_gru_rec_base_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2   --model=SASRec --num_epochs=200 > logs_macr/ml-1m_sas_rec_base_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=False --debias=True --adversarial=False --additive_bias=False  > logs_macr/ml-1m_gru_rec_macr_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=SASRec --num_epochs=200 --c0=30 --disentangle=False --debias=True --adversarial=False --additive_bias=False  > logs_macr/ml-1m_sas_rec_macr_$i.log  2>&1 &


#python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=0.8 > logs_macr/ml-1m_gru_rec_dist_r0.8_$i.log  2>&1 &

#python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=0.8 > logs_macr/ml-1m_sas_rec_dist_r0.8_$i.log  2>&1 &
wait_function

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=1.0 > logs_macr/ml-1m_gru_rec_dist_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=1.0 > logs_macr/ml-1m_sas_rec_dist_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --dynamic_pop_int_weight=True > logs_macr/ml-1m_gru_rec_dist_dyn_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --dynamic_pop_int_weight=True > logs_macr/ml-1m_sas_rec_dist_dyn_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=True --additive_bias=False --dynamic_pop_int_weight=True --int_pop_match_loss_w=0.1 --int_loss_w=0 > logs_macr/ml-1m_gru_rec_dist_dyn_popadv_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=True --additive_bias=False --dynamic_pop_int_weight=True --int_pop_match_loss_w=0.1 --int_loss_w=0 > logs_macr/ml-1m_sas_rec_dist_dyn_popadv_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2   --model=SASRec --num_epochs=200 --main_loss=pair > logs_macr/ml-1m_sas_rec_base_pair_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=SASRec --num_epochs=200 --c0=30 --disentangle=False --debias=True --adversarial=False --additive_bias=False --main_loss=pair  > logs_macr/ml-1m_sas_rec_macr_pair_$i.log  2>&1 &

python main.py --dataset=ml-1m --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=200 --dropout_rate=0.2 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=1.0 --pop_match_tower=False --main_loss=pair > logs_macr/ml-1m_sas_rec_dist_nopopmatch_pair_$i.log  2>&1 &

wait_function


python main.py --dataset=Electronics --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=1.0 --pop_match_tower=False > logs_macr/Electronics_gru_rec_dist_nopopmatch_$i.log  2>&1 &

python main.py --dataset=Electronics --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=1.0 --pop_match_tower=False > logs_macr/Electronics_sas_rec_dist_nopopmatch_$i.log  2>&1 &

python main.py --dataset=Electronics --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5   --model=GRU4Rec --num_epochs=200 > logs_macr/Electronics_gru_rec_base_$i.log  2>&1 &

python main.py --dataset=Electronics --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5   --model=SASRec --num_epochs=200 > logs_macr/Electronics_sas_rec_base_$i.log  2>&1 &

python main.py --dataset=Electronics --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=False --debias=True --adversarial=False --additive_bias=False  > logs_macr/Electronics_gru_rec_macr_$i.log  2>&1 &

python main.py --dataset=Electronics --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=False --debias=True --adversarial=False --additive_bias=False  > logs_macr/Electronics_sas_rec_macr_$i.log  2>&1 &

#python main.py --dataset=Electronics --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=0.8 > logs_macr/Electronics_gru_rec_dist_r0.8_$i.log  2>&1 &

#python main.py --dataset=Electronics --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=0.8 > logs_macr/Electronics_sas_rec_dist_r0.8_$i.log  2>&1 &
wait_function

python main.py --dataset=Electronics --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=1.0 > logs_macr/Electronics_gru_rec_dist_$i.log  2>&1 &

python main.py --dataset=Electronics --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=1.0 > logs_macr/Electronics_sas_rec_dist_$i.log  2>&1 &

python main.py --dataset=Electronics --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --dynamic_pop_int_weight=True > logs_macr/Electronics_gru_rec_dist_dyn_$i.log  2>&1 &

python main.py --dataset=Electronics --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --dynamic_pop_int_weight=True > logs_macr/Electronics_sas_rec_dist_dyn_$i.log  2>&1 &

python main.py --dataset=Electronics --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=True --additive_bias=False --dynamic_pop_int_weight=True --int_pop_match_loss_w=0.1 --int_loss_w=0 > logs_macr/Electronics_gru_rec_dist_dyn_popadv_$i.log  2>&1 &

python main.py --dataset=Electronics --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=True --additive_bias=False --dynamic_pop_int_weight=True --int_pop_match_loss_w=0.1 --int_loss_w=0 > logs_macr/Electronics_sas_rec_dist_dyn_popadv_$i.log  2>&1 &

python main.py --dataset=Electronics --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5   --model=SASRec --num_epochs=200 --main_loss=pair > logs_macr/Electronics_sas_rec_base_pair_$i.log  2>&1 &

python main.py --dataset=Electronics --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=False --debias=True --adversarial=False --additive_bias=False --main_loss=pair  > logs_macr/Electronics_sas_rec_macr_pair_$i.log  2>&1 &

python main.py --dataset=Electronics --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=1.0 --pop_match_tower=False --main_loss=pair > logs_macr/Electronics_sas_rec_dist_nopopmatch_pair_$i.log  2>&1 &
wait_function


python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=1.0 --pop_match_tower=False > logs_macr/steam_gru_rec_dist_nopopmatch_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=1.0 --pop_match_tower=False > logs_macr/steam_sas_rec_dist_nopopmatch_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5   --model=GRU4Rec --num_epochs=200 > logs_macr/steam_gru_rec_base_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5   --model=SASRec --num_epochs=200 > logs_macr/steam_sas_rec_base_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=False --debias=True --adversarial=False --additive_bias=False  > logs_macr/steam_gru_rec_macr_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=False --debias=True --adversarial=False --additive_bias=False  > logs_macr/steam_sas_rec_macr_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --dynamic_pop_int_weight=True > logs_macr/steam_gru_rec_dist_dyn_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --dynamic_pop_int_weight=True > logs_macr/steam_sas_rec_dist_dyn_$i.log  2>&1 &
wait_function

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=1.0 > logs_macr/steam_gru_rec_dist_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=1.0 > logs_macr/steam_sas_rec_dist_$i.log  2>&1 &

#python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=0.8 > logs_macr/steam_gru_rec_dist_r0.8_$i.log  2>&1 &

#python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=0.8 > logs_macr/steam_sas_rec_dist_r0.8_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=True --additive_bias=False --dynamic_pop_int_weight=True --int_pop_match_loss_w=0.1 --int_loss_w=0 > logs_macr/steam_gru_rec_dist_dyn_popadv_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=True --additive_bias=False --dynamic_pop_int_weight=True --int_pop_match_loss_w=0.1 --int_loss_w=0 > logs_macr/steam_sas_rec_dist_dyn_popadv_$i.log  2>&1 &
python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5   --model=SASRec --num_epochs=200 --main_loss=pair > logs_macr/Steam_sas_rec_base_pair_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=False --debias=True --adversarial=False --additive_bias=False --main_loss=pair  > logs_macr/Steam_sas_rec_macr_pair_$i.log  2>&1 &

python main.py --dataset=Steam --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=1.0 --pop_match_tower=False --main_loss=pair > logs_macr/Steam_sas_rec_dist_nopopmatch_pair_$i.log  2>&1 &

wait_function


python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=1.0 --pop_match_tower=False > logs_macr/video_gru_rec_dist_nopopmatch_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=1.0 --pop_match_tower=False > logs_macr/video_sas_rec_dist_nopopmatch_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5   --model=GRU4Rec --num_epochs=200 > logs_macr/video_gru_rec_base_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5   --model=SASRec --num_epochs=200 > logs_macr/video_sas_rec_base_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=False --debias=True --adversarial=False --additive_bias=False  > logs_macr/video_gru_rec_macr_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=False --debias=True --adversarial=False --additive_bias=False  > logs_macr/video_sas_rec_macr_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --dynamic_pop_int_weight=True > logs_macr/video_gru_rec_dist_dyn_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --dynamic_pop_int_weight=True > logs_macr/video_sas_rec_dist_dyn_$i.log  2>&1 &
wait_function

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=1.0 > logs_macr/video_gru_rec_dist_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=1.0 > logs_macr/video_sas_rec_dist_$i.log  2>&1 &

#python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=0.8 > logs_macr/video_gru_rec_dist_r0.8_$i.log  2>&1 &

#python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=0.8 > logs_macr/video_sas_rec_dist_r0.8_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=GRU4Rec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=True --additive_bias=False --dynamic_pop_int_weight=True --int_pop_match_loss_w=0.1 --int_loss_w=0 > logs_macr/video_gru_rec_dist_dyn_popadv_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=True --additive_bias=False --dynamic_pop_int_weight=True --int_pop_match_loss_w=0.1 --int_loss_w=0 > logs_macr/video_sas_rec_dist_dyn_popadv_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5   --model=SASRec --num_epochs=200 --main_loss=pair > logs_macr/Video_sas_rec_base_pair_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=False --debias=True --adversarial=False --additive_bias=False --main_loss=pair  > logs_macr/Video_sas_rec_macr_pair_$i.log  2>&1 &

python main.py --dataset=Video --train_dir=default --hidden_units=50 --u_hidden_units=6 --maxlen=50 --dropout_rate=0.5 --model=SASRec --num_epochs=200 --c0=30 --disentangle=True --debias=True --adversarial=False --additive_bias=False --c1=1.0 --pop_match_tower=False --main_loss=pair > logs_macr/Video_sas_rec_dist_nopopmatch_pair_$i.log  2>&1 &

wait_function

i=$[$i+1]
done

find logs_macr -type f -name "*log" -exec awk '{s=$0};END{print FILENAME,s}' {} \; | sort