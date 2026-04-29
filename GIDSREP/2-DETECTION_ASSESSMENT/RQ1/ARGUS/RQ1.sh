#RQ1

#best Margin Parameter modify line 184  of ./argus/model/argus.py
python main.py --dataset OPTC --hidden 64 -z 32 --rnn GRU --delta 0.1 --lr 0.005 --fpweight 0.55 --epoch 100 --patience 10   

python main.py --dataset OPTC --hidden 64 -z 32 --rnn GRU --delta 0.5 --lr 0.005 --fpweight 0.55 --epoch 100 --patience 10
python main.py --dataset OPTC --hidden 64 -z 32 --rnn GRU --delta 1 --lr 0.005 --fpweight 0.55 --epoch 100 --patience 10
python main.py --dataset OPTC --hidden 64 -z 32 --rnn GRU --delta 1.5 --lr 0.005 --fpweight 0.55 --epoch 100 --patience 10

python main.py --dataset OPTC --hidden 64 -z 32 --rnn GRU --delta 0.1 --lr 0.0005 --fpweight 0.55 --epoch 100 --patience 10
python main.py --dataset OPTC --hidden 64 -z 32 --rnn GRU --delta 0.1 --lr 0.01 --fpweight 0.55 --epoch 100 --patience 10
python main.py --dataset OPTC --hidden 64 -z 32 --rnn GRU --delta 0.1 --lr 0.01 --fpweight 0.55 --epoch 100 --patience 10

python main.py --dataset OPTC --hidden 32 -z 16 --rnn GRU --delta 0.1 --lr 0.005 --fpweight 0.55 --epoch 100 --patience 10
python main.py --dataset OPTC --hidden 16 -z 8 --rnn GRU --delta 0.1 --lr 0.005 --fpweight 0.55 --epoch 100 --patience 10

python main.py --dataset OPTC --hidden 64 -z 32 --rnn LSTM --delta 0.1 --lr 0.005 --fpweight 0.55 --epoch 100 --patience 10

#Margin Parameter modify line 184  of ./argus/model/argus.py 0.3,0.5,0.7,0.9
#python main.py --dataset OPTC --hidden 64 -z 32 --rnn GRU --delta 0.1 --lr 0.005 --fpweight 0.55 --epoch 100 --patience 10 
#python main.py --dataset OPTC --hidden 64 -z 32 --rnn GRU --delta 0.1 --lr 0.005 --fpweight 0.55 --epoch 100 --patience 10 
#python main.py --dataset OPTC --hidden 64 -z 32 --rnn GRU --delta 0.1 --lr 0.005 --fpweight 0.55 --epoch 100 --patience 10 