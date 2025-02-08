Setting up the environment and installing the necessary packages (dependencies)

1. conda --version
2. conda activate base
3. conda activate masaenv
4. pip install -U openmim
5. mim install openmim
6. mim install mmengine
7. pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html 
8. pip install git+https://github.com/open-mmlab/mmdetection.git@v3.3.0
9. pip install -r masa/requirements.txt
10. pip install huggingface_hub
11. pip install ultralytics
12. nltk.download('all')  # -- optional
13. pip install ipywidgets # -- if an error occurs (optional)
