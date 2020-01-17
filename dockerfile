FROM pytorch/pytorch
RUN pip install --upgrade pip
RUN pip install pytorch-ignite tqdm jupyterlab
