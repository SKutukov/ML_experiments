python train.py \
--current_epoch 0 \
--min_reward -200 \
--epochs_count 100000 \
--time_limit 5 \
--train_model hybriteModel \
--restore_path res/last_weight/2/LunarLander-v2.ckpt \
--load_version 0 \
--render 
# --no-render
