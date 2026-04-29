#RQ1

#best
python main.py --dataset OPTC --hidden 32 -z 16 --rnn GRU -d 0.1 --lr 0.01 --fpweight 0.6 --epoch 100 --patience 10 -t 5

python main.py --dataset OPTC --hidden 32 -z 16 --rnn GRU -d 0.1 --lr 0.01 --fpweight 0.6 --epoch 100 --patience 10 -t 5
python main.py --dataset OPTC --hidden 32 -z 16 --rnn GRU -d 0.5 --lr 0.01 --fpweight 0.6 --epoch 100 --patience 10 -t 5
python main.py --dataset OPTC --hidden 32 -z 16 --rnn GRU -d 1 --lr 0.01 --fpweight 0.6 --epoch 100 --patience 10 -t 5
python main.py --dataset OPTC --hidden 32 -z 16 --rnn GRU -d 1.5 --lr 0.01 --fpweight 0.6 --epoch 100 --patience 10 -t 5


python main.py --dataset OPTC --hidden 32 -z 16 --rnn GRU -d 1 --lr 0.005 --fpweight 0.6 --epoch 100 --patience 10 -t 5
python main.py --dataset OPTC --hidden 32 -z 16 --rnn GRU -d 1 --lr 0.05 --fpweight 0.6 --epoch 100 --patience 10 -t 5


python main.py --dataset OPTC --hidden 64 -z 32 --rnn GRU -d 1 --lr 0.01 --fpweight 0.6 --epoch 100 --patience 10 -t 5
python main.py --dataset OPTC --hidden 16 -z 8 --rnn GRU -d 1 --lr 0.01 --fpweight 0.6 --epoch 100 --patience 10 -t 5

python main.py --dataset OPTC --hidden 32 -z 16 --rnn LSTM -d 1 --lr 0.01 --fpweight 0.6 --epoch 100 --patience 10 -t 5
python main.py --dataset OPTC --hidden 32 -z 16 --rnn NONE -d 1 --lr 0.01 --fpweight 0.6 --epoch 100 --patience 10 -t 5

