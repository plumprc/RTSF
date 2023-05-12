<!-- ETTh1 -->

python -u run.py --is_training 1 --data_path ETTh1.csv --data ETTh1 --features M --channel 7 --itr 1 --batch_size 128 --learning_rate 0.005 --model Linear --rev --seq_len 336 --pred_len 96 --gpu 3

<!-- ETTm1 -->

python -u run.py --is_training 1 --data_path ETTm1.csv --data ETTm1 --features M --channel 7 --itr 1 --batch_size 128 --learning_rate 0.005 --model Linear --rev --seq_len 336 --pred_len 96 --gpu 3

<!-- ETTh2 -->

python -u run.py --is_training 1 --data_path ETTh2.csv --data ETTh2 --features M --channel 7 --itr 1 --batch_size 256 --learning_rate 0.005 --model Linear --rev --seq_len 336 --pred_len 96 --gpu 3

<!-- ETTm2 -->

python -u run.py --is_training 1 --data_path ETTm2.csv --data ETTm2 --features M --channel 7 --itr 1 --batch_size 128 --learning_rate 0.005 --model Linear --rev --seq_len 336 --pred_len 96 --gpu 3

<!-- Weather -->

python -u run.py --is_training 1 --data_path Weather.csv --data custom --features M --channel 21 --itr 1 --batch_size 128 --learning_rate 0.005 --model Linear --rev --seq_len 336 --pred_len 96 --gpu 3

<!-- ECL -->

python -u run.py --is_training 1 --data_path ECL.csv --data custom --features M --channel 321 --itr 1 --learning_rate 0.001 --model Linear --rev --seq_len 336 --pred_len 96 --gpu 3 --drop 0
