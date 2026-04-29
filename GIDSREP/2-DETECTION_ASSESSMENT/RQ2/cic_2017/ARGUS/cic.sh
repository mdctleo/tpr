#lanl Margin Parameter=0.8 modify line 184 of ./argus/model/argus.py 
python main.py --dataset O_cic --delta 15 --lr 0.01 --hidden 32 -z 16 --fpweight 0.46  --epoch 100 --patience 3

#lanl Margin Parameter=0.8 modify line 181 of ./argus/model/argus.py 
python main.py --dataset L_cic_flow --delta 10 --lr 0.05 --hidden 32 -z 16 --fpweight 0.45    --epoch 100 --patience 3
