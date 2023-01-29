<!-- ECL -->

python -u run.py --is_training 1 --data_path ECL.csv --data custom --features M --seq_len 96 --pred_len 96 --channel 321 --itr 1 --learning_rate 0.001 --model Linear --train_epochs 3

<!-- Exchange -->

python -u run.py --is_training 1 --data_path Exchange.csv --data custom --features M --seq_len 96 --pred_len 96 --channel 8 --itr 1 --batch_size 8 --learning_rate 0.0005 --model Linear --rev --norm --train_epochs 2

<!-- ETTh1 -->

python -u run.py --is_training 1 --data_path ETTh1.csv --data ETTh1 --features M --seq_len 96 --pred_len 96 --channel 7 --itr 1 --batch_size 32 --learning_rate 0.005 --model Linear --rev --norm --train_epochs 6
