Questo programma è una implementazione pytorch basata su una CNN che riconosce le emozioni in real time o da un video.

## Dependencies

-Per creare un ambiente di lavoro adeguato:
# $ conda create --name <env>
matplotlib
numpy
opencv-python
pip=20.1.1
python=3.8.3
pytorch=1.5.1
torchvision=0.6.1


##Training
Il modello è stato pre-allenato con dataset del FER-2013 (https://www.kaggle.com/datasets/msambare/fer2013)
Il training è stato eseguito su google colab
Il dataset iniziale viene anche manipolato(Data Augmentation)
__Nel modello presente l'accuratezza è al 65.06% dopo 24 epoche

##Riconoscimento
Per avviare il riconoscimento tramite webcame bisogna eseguire video_capture.py
Qualora si volesse avviare il programma su un determinato video prescaricato è possibile modificando opportunatamente il codice di video_capture.py

##Il programma farà partire il video, mostrando un rettangolo attorno al viso del soggetto inquadrato e l'evoluzione delle emozioni.

DA FARE:
--altri visi DONE
--more grafici DONE
