This is the code to our work entitled A Comparative Study of Language Transformers for Video Question Answering. We used language Transformers to represent both visual and language elements and compared the performance of four different Transformers (BERT, XLNet, RoBERTa and ALBERT).

This work has been published on Neurocomputing entitled "A comparative study of language transformers for video question answering".

You need to install python 3.6, Nvidia driver and cuda toolkit.

To use this code, first download the package first, then download TVQA dataset at http://tvqa.cs.unc.edu/download_tvqa.html (with visual concepts, subtitles, questions and answers) and put it into ./data/raw. 

Then, run src/preprocessing.py for pre-processing. The processed data is shown in ./data/processed.

Next, Install all the packages in requirements.txt. 

To execute, run "python ./bin/run.py". The trained model is saved in ./models

For test, run "python ./src/test.py --model_dir (your models) --mode valid"

Note: To run the code, please keep the command linein the root dir, otherwise an error might happen.

You can modify settings in ./src/config.py to change the configurations in the training, including the model, max_seq_len etc.

We also provide docker environment. 

After downloading this package and the dataset, run

docker build -t (image name) dockerfiles/vqatransformers

to build the container. 

To run the code, use

bash bin/run.sh --gpus=all (image name) python run.py

For validation, use

bash bin/run.sh --gpus=all (image name) python src/test.py --model_dir (your models) --mode valid


