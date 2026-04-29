#RQ2


#Reproduction
#lanl 
python main.py --dataset LANL --hidden 32 -z 16 --rnn GRU -d 0.5 --lr 0.005 --fpweight 0.6 --epoch 100 --patience 5 -t 5

#optc 
python main.py --dataset OPTC --hidden 32 -z 16 --rnn LSTM -d 2.5 --lr 0.005 --fpweight 0.6 --epoch 100 --patience 5 -t 5 

#Replication

#lanl 
python main.py --dataset LANL --hidden 64 -z 32 --rnn NONE -d 3 --lr 0.0005 --fpweight 0.6 --epoch 100 --patience 10 -t 5
#optc 
python main.py --dataset OPTC --hidden 32 -z 16 --rnn GRU -d 0.1 --lr 0.01 --fpweight 0.6 --epoch 100 --patience 10 -t 5

