#RQ3

#Replication

#n9000 without flow Margin Parameter=0.8 modify line 181 of ./argus/model/argus.py 
python main.py --dataset O_n9000 --hidden 32 -z 16 --rnn GRU --delta 2.5 --lr 0.0001 --fpweight 0.44 --epoch 200 --patience 10 

#n9000 with flow Margin Parameter=0.8 modify line 184 of ./argus/model/argus.py 
python main.py --dataset L_n9000 --hidden 32 -z 16 --rnn GRU --delta 7.5 --lr 0.0005 --fpweight 0.46 --epoch 200 --patience 10 

#wizard without flow Margin Parameter=0.8 modify line 199 of ./argus/model/argus.py 
python main.py --dataset O_wizard --hidden 16 -z 8 --rnn GRU --delta 2.5 --lr 0.001 --fpweight 0.48 --epoch 100 --patience 3 

#wizard with flow Margin Parameter=0.8 modify line 202 of ./argus/model/argus.py 
python main.py --dataset L_wizard --hidden 32 -z 16 --rnn GRU --delta 2.5 --lr 0.0001 --fpweight 0.46 --epoch 100 --patience 5 

#oilrig without flow Margin Parameter=0.8 modify line 187 of ./argus/model/argus.py 
python main.py --dataset O_oilrig --hidden 32 -z 16 --rnn GRU --delta 7.5 --lr 0.01 --fpweight 0.48 --epoch 100 --patience 5

#oilrig with flow Margin Parameter=0.8 modify line 190 of ./argus/model/argus.py 
python main.py --dataset L_oilrig --hidden 32 -z 16 --rnn GRU --delta 2.5 --lr 0.0001 --fpweight 0.46 --epoch 100 --patience 3 

#sandworm without flow Margin Parameter=0.8 modify line 193 of ./argus/model/argus.py 
python main.py --dataset O_sandworm --hidden 16 -z 8 --rnn GRU --delta 2.5 --lr 0.0001 --fpweight 0.43 --epoch 100 --patience 3

#sandworm with flow Margin Parameter=0.8 modify line 196 of ./argus/model/argus.py 
python main.py --dataset L_sandworm --hidden 32 -z 16 --rnn GRU --delta 7.5 --lr 0.0001 --fpweight 0.45 --epoch 100 --patience 3 