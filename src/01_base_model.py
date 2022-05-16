import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import tensorflow as tf
import numpy as np
from keras.models import Sequential

STAGE = "Base Model" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)
    # params = read_yaml(params_path)
    
    #get the data 
    mnist=tf.keras.datasets.mnist
    (X_train,y_train),(X_test,y_test)=mnist.load_data()
    X_valid,X_train=X_train[:5000]/255.0,X_train[5000:]/255.0
    y_valid,y_train=y_train[:5000],y_train[5000:]
    X_test=X_test/255.

    #set the seeds
    seed=2021
    tf.random.set_seed(seed)
    np.random.seed(seed)

    #define Layers
    Layers = [
        tf.keras.layers.Flatten(input_shape=[28,28],name="input_layer"),
        tf.keras.layers.Dense(300,name="hidddenlayer1"),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Dense(100,name="hidddenlayer2"),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(10,activation="softmax",name="output_layer")
        ]
    # define the model and compile it
    model=Sequential(Layers)
    LOSS_FUNCTION="sparse_categorical_crossentropy"
    OPTIMIZER="SGD"
    METRICS=["accuracy"]

    model.compile(loss=LOSS_FUNCTION,optimizer=OPTIMIZER,metrics=METRICS)
    model.summary()
    logging.info(model.summary())

    #Train the model
    EPOCS=10
    VALIDATION=(X_valid,y_valid)
    Trained_model=model.fit(X_train,y_train,epochs=EPOCS,validation_data=VALIDATION)

    #save the model 
    path_to_dir=os.path.join("artifacts","saved_models")
    os.makedirs(path_to_dir,exist_ok=True)
    model_file_path=os.path.join(path_to_dir,"base_model.h5")
    model.save(model_file_path)

    #logging info
    logging.info(f"base model is saved st : {model_file_path}")
    logging.info(f">>>>>>> Model evaluation : {model.evaluate(X_test,y_test)}")
    




    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    # args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        #  params_path=parsed_args.params
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e