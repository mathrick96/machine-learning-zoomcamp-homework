# image downloaded from the internet 
FROM svizor/zoomcamp-model:3.11.5-slim

# installing pipenv
RUN pip install pipenv

# creating and changing to the working directory
WORKDIR /app

# copy the important files to install the dependecies
COPY ["Pipfile", "Pipfile.lock", "./"]

# instal the dependencies
RUN pipenv install --system --deploy

# copy the model files
COPY ["dv.bin", "model1.bin", "score_customer.py", "./"]

# expose the port
EXPOSE 9696

# connect the 
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "score_customer:app" ]