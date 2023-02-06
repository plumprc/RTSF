<!-- ECL -->

python -u run.py --is_training 1 --data_path ECL.csv --data custom --features M --channel 321 --itr 1 --learning_rate 0.001 --model Flow --train_epochs 6 --seq_len 96 --pred_len 96 --gpu 3

<!-- Exchange -->

python -u run.py --is_training 1 --data_path Exchange.csv --data custom --features M --channel 8 --itr 1 --batch_size 8 --learning_rate 0.0005 --model Flow --rev --norm --train_epochs 4 --seq_len 96 --pred_len 96 --gpu 3

<!-- ETTh1 -->

python -u run.py --is_training 1 --data_path ETTh1.csv --data ETTh1 --features M --channel 7 --itr 1 --batch_size 32 --learning_rate 0.005 --model Flow --rev --norm --train_epochs 6 --seq_len 96 --pred_len 96 --gpu 3
