#RQ2



#Reproduction
#lanl Margin Parameter=0.8 modify line 181 of ./argus/model/argus.py 
python main.py --dataset LANL --hidden 32 -z 16 --rnn GRU --delta 1 --lr 0.01 --fpweight 0.6 --epoch 100 --patience 3 

#optc Margin Parameter=0.8 modify line 184 of ./argus/model/argus.py 
python main.py --dataset OPTC --hidden 32 -z 16 --rnn GRU --delta 0.1 --lr 0.005 --fpweight 0.6 --epoch 100 --patience 10 

#Replication
#lanl Margin Parameter=0.8 modify line 181 of ./argus/model/argus.py 
python main.py --dataset LANL --hidden 32 -z 16 --rnn GRU --delta 1 --lr 0.005 --fpweight 0.6 --epoch 100 --patience 10 

#optc Margin Parameter=0.1 modify line 181 of ./argus/model/argus.py 
python main.py --dataset OPTC --hidden 64 -z 32 --rnn GRU --delta 0.1 --lr 0.005 --fpweight 0.55 --epoch 100 --patience 10 