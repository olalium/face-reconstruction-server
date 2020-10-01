FROM python:3.6

WORKDIR /usr/src/app

# need this for dlib installation later
RUN pip install cmake

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-u","./predictor.py" ]