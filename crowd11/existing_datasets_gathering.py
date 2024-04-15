import csv
import os
from shutil import copyfile


### Folder in which the selected videos of existing datasets will be copied
outputFolder = "./VOI/"


### Folder in which the existing datasets are
existingDatasetsFolder = "./existing_datasets"

### UMN DATASET
### can be found at http://mha.cs.umn.edu/proj_events.shtml#crowd
umnFolder = existingDatasetsFolder + "/umn/"

### AGORASET DATASET
### can be found at https://www.sites.univ-rennes2.fr/costel/corpetti/agoraset/Site/AGORASET.html
agorasetFolder = existingDatasetsFolder + "/agoraset/"

### PETS DATASET
### can be found at http://www.cvg.reading.ac.uk/PETS2009/a.html#s3
petsFolder = existingDatasetsFolder + "/pets/"

### HOCKEY FIGHT DATASET
### can be found at http://visilab.etsii.uclm.es/personas/oscar/FightDetection/
hockeyFolder = existingDatasetsFolder + "/hockey/"

### MOVIES DATASET
### can be found at http://visilab.etsii.uclm.es/personas/oscar/FightDetection/
peliculasFolder = existingDatasetsFolder + "/peliculas/"

### CUHK DATASET
### can be found at http://www.ee.cuhk.edu.hk/~jshao/CUHKcrowd_files/cuhk_crowd_dataset.htm
cuhkFolder = existingDatasetsFolder + "/cuhk/"

### WWW DATASET
### can be found at http://www.ee.cuhk.edu.hk/~jshao/WWWCrowdDataset.html
wwwFolder = existingDatasetsFolder + "/www/"

### WORLDEXPO'10 CROWD COUNTING DATASET
### can be found at http://www.ee.cuhk.edu.hk/~xgwang/expo.html
shanghaiFolder = existingDatasetsFolder + "/Shanghai_dataset/"

### VIOLENT-FLOWS DATASET
### can be found at http://www.openu.ac.il/home/hassner/data/violentflows/
violentFolder = existingDatasetsFolder + "/violent_flow/"

with open('./existing_datasets_urls.csv', 'r') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=';')
    counter = 0
    for row in spamreader:
            print(counter, row)
            source = row[0]
            output_name = row[1]

            # create origin folder if not exists
            if not os.path.exists(outputFolder + source):
                os.makedirs(outputFolder + source)

            if (source == "www" and os.path.exists(wwwFolder)):
                copyfile(wwwFolder + output_name, outputFolder + source + '/' + output_name)
            
            if (source == "cuhk" and os.path.exists(cuhkFolder)):
                copyfile(cuhkFolder + output_name, outputFolder + source + '/' + output_name)

            if (source == "umn" and os.path.exists(umnFolder)):
                copyfile(umnFolder + output_name, outputFolder + source + '/' + output_name)

            if (source == "violent_flow" and os.path.exists(violentFolder)):
                copyfile(violentFolder + output_name, outputFolder + source + '/' + output_name)

            if (source == "agoraset" and os.path.exists(agorasetFolder)):
                copyfile(agorasetFolder + output_name, outputFolder + source + '/' + output_name)

            if (source == "pets" and os.path.exists(petsFolder)):
                copyfile(petsFolder + output_name, outputFolder + source + '/' + output_name)

            if (source == "hockey" and os.path.exists(hockeyFolder)):
                copyfile(hockeyFolder + output_name, outputFolder + source + '/' + output_name)

            if (source == "shanghai" and os.path.exists(shanghaiFolder)):
                copyfile(shanghaiFolder + output_name, outputFolder + source + '/' + output_name)

            if (source == "peliculas" and os.path.exists(peliculasFolder + "fights/")):
                copyfile(peliculasFolder + "fights/" + output_name, outputFolder + source + '/' + output_name)

            

            counter += 1
            
if not os.path.exists(outputFolder + "pond5"):
    os.makedirs(outputFolder + "pond5")
if not os.path.exists(outputFolder + "youtube"):
    os.makedirs(outputFolder + "youtube")
if not os.path.exists(outputFolder + "gettyimages"):
    os.makedirs(outputFolder + "gettyimages")
    
