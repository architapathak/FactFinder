# FactFinder: Django Web App For End-to-End Fact-Checking Pipeline
The web app is based on the following pipeline:

![Website](https://user-images.githubusercontent.com/25678184/111694946-e7806780-8808-11eb-9ecc-c35c4ea24ee8.png)

## Setting Up Python Environment
To reproduce the web app on your machine, we recommend setting up a python3 virtual environment. To setup virtual environment, first install `virtualenv` using pip3

```
pip3 install virtualenv
```

Next, to create python3 virual enverionment named `newenv`, run following the command

```
virtualenv -p python3 venv
```

Following command is used to activate the `newenv` environmen

```
source newenv/bin/activate
```
You can now work with this environment to install dependencies for the web app. To exit the virual environment, simply run `deactivate` command from anywhere.

## Installing/Downloading Dependencies
Once inside the project directory on virtual environment, the following command can be used to install dependencies as mentioned in **requirements.txt** file

```
pip3 install -r requirements.txt
```

The webapp also requires GoogleNews word2vec vectors for Clickbait predictions. The embeddings can be downloaded and copied to correct location by using the following command:

```
wget -P /modules/models/ -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
```
Note that the size of this file is 1.5G and will take around 5 minutes to download.

Finally, you would also require nltk `punkt` and `stopwords` packages that can be downloaded by using the following commands:
```
nltk.download('punkt')
nltk.download('stopwords')
```

## Run Django Web App