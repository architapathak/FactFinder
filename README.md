# FactFinder: Django Web App For End-to-End Fact-Checking Pipeline
This repository contains code and models for web app construction using Django framework. A demo of this app was presented at Disinformation Challenge organized by 2020 SBP-BRiMS as a part of Challenge Winner paper *Disinformation - Analysis and Identification*

The web app is based on the following pipeline:

![Website](https://user-images.githubusercontent.com/25678184/111694946-e7806780-8808-11eb-9ecc-c35c4ea24ee8.png)

## Setting Up Python Environment
To reproduce the web app on your machine, we recommend setting up a python3 virtual environment. To setup virtual environment, first install `virtualenv` using pip3. **Note that dependencies are based on python version >= 3.7 and torch >= 1.3**

```
pip3 install virtualenv
```

Next, to create python3 virual enverionment named `newenv`, run following the command

```
virtualenv -p python3 newenv
```

Following command is used to activate the `newenv` environment

```
source newenv/bin/activate
```
You can now clone the repository within this environment and install dependencies for the web app. To exit the virual environment, simply run `deactivate` command from anywhere.

## Installing/Downloading Dependencies
**Note that models stored in location *modules/models/* have total size 835M. Please download them manually or use git lfs (https://git-lfs.github.com/)**

Once inside the project directory on virtual environment, the following command can be used to install dependencies as mentioned in **requirements.txt** file

```
pip3 install -r requirements.txt
```

For query formulation module, you would require bert_sklearn which can be installed by using the following steps:

```
git clone -b master https://github.com/charles9n/bert-sklearn
cd bert-sklearn
pip install .
```

The webapp also requires GoogleNews word2vec vectors for Clickbait predictions. The embeddings can be downloaded and copied to correct location (***modules/models/***) by using the following command:

```
wget -P modules/models/ -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
```
Note that the size of this file is 1.5G and will take around 5 minutes to download.

Finally, you would also require nltk `punkt` and `stopwords` packages that can be downloaded by using:
```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Run Django Web App
From the FactFinder project directory, first, run the command for Django migration:

```
python manage.py migrate
```
To start the web app on `127.0.0.1:8000`

```
python manage.py runserver
```
