import requests
import urllib.request
import time
from bs4 import BeautifulSoup
from selenium import webdriver
import json
from os import listdir
from os.path import isfile, join

onlyfiles = [f.split(".")[0] for f in listdir("images/") if isfile(join("images/", f))]

i = 0
while (True):
    if (i > 247):
        break;
    response = requests.get("https://api.dimu.org/api/solr/select?q=*&fq=artifact.producer:Edvard%20Munch&wt=json&api.key=demo&rows=10&start=" + str(i))
    images =  json.loads(response.text)['response']['docs']
    
    for image in images:
        if (not image["artifact.defaultMediaIdentifier"] in onlyfiles):
            loaded  = requests.get("https://dms01.dimu.org/image/" + image["artifact.defaultMediaIdentifier"] +"?dimension=max").content
            with open("images/"+ image["artifact.defaultMediaIdentifier"] + '.jpg', 'wb') as handler:
                handler.write(loaded)
    i += 9
 

""" 
DRIVER_PATH = '/Users/jdam/Code/glup-dings/chromedriver'
driver = webdriver.Chrome(executable_path=DRIVER_PATH)
driver.get('https://www.nasjonalmuseet.no/en/collection/producer/56154/edvard-munch')

 
elementName = driver.find_element(by="")
 """