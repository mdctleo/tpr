# DATA PROCESSING MODULE



# LANL

Data set concatenation: https://csr.lanl.gov/data/cyber1/

## argus

Fill lines 6-9 in the split_lanl_argus.py file

```python
RED = '' # Location of redteam.txt
SRC = '' # Location of auth.txt
DST = '' # Directory to save output files to
SRC_DIR = '' # Directory of flows.txt, auth.txt
```

then,

```cmd
python split_lanl_argus.py
```

## euler & vgrnn

Fill in lines 13-15 of the split_lanl_euler.py file

```python
RED = '' # Location of redteam.txt
SRC = '' # Location of auth.txt
DST = '' # Directory to save output files to
```

then,

```cmd
python split_lanl_euler.py
```

## PIKACHU

Fill lines 9-12 in the split_lanl_pikachu.py file

```python
RED = '' # Location of redteam.txt
SRC = '' # Location of auth.txt
DST = 'lanl_3600.csv' # output files 
snapshot= 3600  #  length of snapshot (s)for RQ 2 it should be set 3600
```

then,

```cmd
python split_lanl_pikachu.py
```



# OPTC

Link to official dataset:https://github.com/FiveDirections/OpTC-data

Experiment with database connections (from the argus code base)：https://drive.google.com/drive/folders/1pTU-ZcyJbzoB1FuvujXe-ynaUy8O-PVD?usp=sharing

## argus & euler & vgrnn

Fill lines 7-9 in the split_optc.py file

```python
RED = '' # Location of redteam.txt
SRC = '' # Location of auth.txt
DST = '' # Directory to save output files to
```

then,

```cmd
python split_optc.py
```

## pikachu

Fill lines 13-15 in the split_optc_pikachu.py file

```python
SRC = 'auth_optc.txt' # Location of auth_optc.txt
DST = 'optc_3600.csv' # output file optc_360.csv,optc_1800.csv
snapshot= 3600  #  length of snapshot (s)  
#for RQ1 it should be set 360,1800 and 3600
#for RQ2 it should be set 360 and 3600
```

then,

```cmd
python split_optc_pikachu.py
```



# cic-2017

Link to official dataset:https://www.unb.ca/cic/datasets/ids-2017.html

Run **process_cic_2017.ipynb**



# 0501(real_world)

Link to official dataset:https://zenodo.org/records/15616158?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjdmNzZkMDc0LWU3NzItNGVjZC05YmZmLTFmOWY1MGEwMmFkNCIsImRhdGEiOnt9LCJyYW5kb20iOiIyN2I5NDM2ODkzYWY5YWExMTgxNjAyODQ1YjEwZmQxOSJ9.zI_ZJr491MvcJG105xqIyQGLc2thRI-oMfkVHIy3ykIyrhcGZzi5JdiZAhkAZhEtFSKLV5Pno6Td3Q9mYmYNAQ

**Put the files in the compressed package into the folder "./0501/"**

## with flow features (argus)

```cmd
python split_0501_flow.py --input label_n9000.log --name n9000_flow
python split_0501_flow.py --input label_oilrig.log --name oilrig_flow
python split_0501_flow.py --input label_sandworm.log --name sandworm_flow
python split_0501_flow.py --input label_wizard_spider.log --name wizard_spider_flow
```

--input  Data set log address

--name  Data set name

## without flow features(argus,euler,vgrnn)

```cmd
python split_0501.py --input label_n9000.log --name n9000
python split_0501.py --input label_oilrig.log --name oilrig
python split_0501.py --input label_sandworm.log --name sandworm
python split_0501.py --input label_wizard_spider.log --name wizard_spider
```

--input  Data set log address

--name  Data set name

## pikachu

```cmd
python split_0501_pikachu.py --input label_n9000.log --name n9000 --time 5
python split_0501_pikachu.py --input label_oilrig.log --name oilrig --time 5
python split_0501_pikachu.py --input label_sandworm.log --name sandworm --time 5
python split_0501_pikachu.py --input label_wizard_spider.log --name wizard_spider --time 5
```

--input  Data set log address

--name  Data set name

--time  time of snapshot  (min)

## Anomal_E

```cmd
python split_0501_anomal.py --input label_n9000.log --name n9000 
python split_0501_anomal.py --input label_oilrig.log --name oilrig
python split_0501_anomal.py --input label_sandworm.log --name sandworm 
python split_0501_anomal.py --input label_wizard_spider.log --name wizard_spider 
```

--input  Data set log address

--name  Data set name


