FROM jupyter/scipy-notebook

COPY requirements.txt .

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD echo "Running Docker Image"
