
##best
python main.py --dataset optc --input dataset/optc_360.csv --trainwin 1469 --dimensions 64 --alpha 0.001 --support 10

#snapshot
python main.py --dataset optc --input dataset/optc_1800.csv --trainwin 296 --dimensions 64 --alpha 0.001 --support 10
python main.py --dataset optc --input dataset/optc_3600.csv --trainwin 150 --dimensions 64 --alpha 0.001 --support 10


#embedding
python main.py --dataset optc --input dataset/optc_360.csv --trainwin 1469 --dimensions 100 --alpha 0.001 --support 10
python main.py --dataset optc --input dataset/optc_360.csv --trainwin 1469 --dimensions 200 --alpha 0.001 --support 10

#learning_rate
python main.py --dataset optc --input dataset/optc_360.csv --trainwin 1469 --dimensions 64 --alpha 0.005 --support 10
python main.py --dataset optc --input dataset/optc_360.csv --trainwin 1469 --dimensions 64 --alpha 0.01 --support 10

#neighbor number
python main.py --dataset optc --input dataset/optc_360.csv --trainwin 1469 --dimensions 64 --alpha 0.001 --support 15
python main.py --dataset optc --input dataset/optc_360.csv --trainwin 1469 --dimensions 64 --alpha 0.001 --support 20

