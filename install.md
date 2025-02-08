# Setting up the environment and installing the necessary packages (dependencies)

## Check conda version
```sh
conda --version
```

## Activate the base environment
```sh
conda activate base
```

## Activate the masaenv environment
```sh
conda activate masaenv
```

## Install openmim
```sh
pip install -U openmim
```

## Install mim
```sh
mim install openmim
```

## Install mmengine
```sh
mim install mmengine
```

## Install mmcv
```sh
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html 
```

## Install mmdetection
```sh
pip install git+https://github.com/open-mmlab/mmdetection.git@v3.3.0
```

## Install required packages from masa/requirements.txt
```sh
pip install -r masa/requirements.txt
```

## Install huggingface_hub
```sh
pip install huggingface_hub
```

## Install ultralytics
```sh
pip install ultralytics
```

## Download all nltk data (optional)
```sh
nltk.download('all')
```

## Install ipywidgets (if an error occurs, optional)
```sh
pip install ipywidgets
```
