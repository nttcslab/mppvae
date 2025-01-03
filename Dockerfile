FROM tensorflow/tensorflow:2.12.0-gpu-jupyter

RUN apt update

RUN pip install jupyterlab==4.0.0 scipy==1.10.1 tqdm tensorflow_probability==0.20.1 pandas==2.0.1 scikit-learn==1.2.2 seaborn==0.12.2 statsmodels==0.14.1

CMD jupyter lab --notebook-dir=/tf --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token="'0000'"
