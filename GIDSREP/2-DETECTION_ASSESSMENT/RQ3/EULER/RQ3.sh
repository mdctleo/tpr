#RQ3
#Replication


python main.py --dataset n9000 --hidden 32 -z 16 --rnn GRU -d 2.5 --lr 0.01 --fpweight 0.45 --epoch 100 --patience 5 -t 5
 
python main.py --dataset oilrig --hidden 32 -z 16 --rnn GRU -d 2.5 --lr 0.005 --fpweight 0.48 --epoch 100 --patience 10 -t 5

python main.py --dataset wizard --hidden 32 -z 16 --rnn GRU -d 2.5 --lr 0.005 --fpweight 0.47 --epoch 100 --patience 10 -t 5

python main.py --dataset sandworm --hidden 32 -z 16 --rnn GRU -d 2.5 --lr 0.005 --fpweight 0.42 --epoch 100 --patience 10 -t 5

