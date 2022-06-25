FROM jupyter/scipy-notebook
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

CMD echo "Running Docker Image"
