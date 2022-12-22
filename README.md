# homework4


## Installation guide

Clone repository 
```shell
git clone https://github.com/marina-shesha/homework4.git
```
Download requirements
```shell
pip install -r homework4/requirements.txt
```

## Training

Run train.py with base_config.json

```shell
%run -i homework4/train.py --config homework4/hw_nv/configs/base_config.json
```
## Test 

Download checkpount and config

```shell
pip install gdown==4.5.4 --no-cache-dir
gdown https://drive.google.com/u/0/uc?id=1Jgg0OF7JFGX2cpwcOjoxZE_zPsWtFt0L&export=download
gdown https://drive.google.com/u/0/uc?id=12D-9VOVkibYevXPOuydfFQ8DVkKUpJts&export=download
```

Run test.py file to get audio preductions

```shell
%run -i homework4/test.py \
--resume /content/checkpoint-epoch68.pth\
--config /content/homework4/hw_nv/configs/base_config.json\
-o test_clean_out.json
```
Audio preductions will be located in dir "test_results". Use  

```shell
from IPython import display
display.display(display.Audio(audio_path))
```
for visualization audio.


You can see usage of this this instruction in ```DLA4.ipynb```
