FROM python:3.9 

COPY ./ . 

RUN pip install librosa seaborn pandas jupyter tensorflow