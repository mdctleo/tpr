#best
python VGRNN.py -n OPTC -s 0.5 -dim 32 -l 0.005 -p 10

#learning rate
python VGRNN.py -n OPTC -s 0.5 -dim 32 -l 0.001 -p 10
python VGRNN.py -n OPTC -s 0.5 -dim 32 -l 0.01 -p 10

#snapshot
python VGRNN.py -n OPTC -s 0.1 -dim 32 -l 0.005 -p 10
python VGRNN.py -n OPTC -s 0.5 -dim 32 -l 0.005 -p 10
python VGRNN.py -n OPTC -s 1 -dim 32 -l 0.005 -p 10
python VGRNN.py -n OPTC -s 1.5 -dim 32 -l 0.005 -p 10

#dim
python VGRNN.py -n OPTC -s 0.5 -dim 64 -l 0.005 -p 10
python VGRNN.py -n OPTC -s 0.5 -dim 16 -l 0.005 -p 10
