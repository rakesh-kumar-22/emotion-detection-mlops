#Base
FROM python:3.10

#workdirectory
WORKDIR /app

#copy
COPY flask_app/ /app/

COPY models/vectorizer.pkl /app/models/vectorizer.pkl
#run
RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords wordnet omw-1.4
#port expose
EXPOSE 5000
#command
CMD ["gunicorn","-b","0.0.0.0:5000","app:app"]