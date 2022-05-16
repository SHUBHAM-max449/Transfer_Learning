import argparse
import os
import shutil
from tkinter import Y
from turtle import update
from pkg_resources import add_activation_listener
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
import random
import numpy as np
import tensorflow as tf
import io


STAGE = "Transfer Learning" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def update_lables(list_of_lables):
    for ind,label in enumerate(list_of_lables):
        list_of_lables[ind]=np.where(label%2==0,1,0)
    return list_of_lables

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

    y_train_bin,y_test_bin,y_valid_bin=update_lables([y_train,y_test,y_valid])

     #set the seeds
    seed=2021
    tf.random.set_seed(seed)
    np.random.seed(seed)

    #load model 
    path_to_model=os.path.join("artifacts","saved_models","base_model.h5")
    base_model=tf.keras.models.load_model(path_to_model)

    # frezee trained model weights
    for layer in base_model.layers[:-1]:
        layer.trainable=False
        print(f"trainable layer {layer.name} : {layer.trainable}")

    base_layer=base_model.layers[:-1]
    new_model=tf.keras.models.Sequential(base_layer)
    new_model.add(tf.keras.layers.Dense(1,activation="sigmoid",name="output_layer"))
    def log_model_summaty(model):
        with io.StringIO() as stream:
            model.summary(print_fn= lambda x: stream.write(f"{x}\n"))
            summary_str =stream.getvalue()
        return summary_str

    logging.info(f"transfer model summary : {log_model_summaty(new_model)}")
    # Training model
    LOSS_FUNCTION="binary_crossentropy"
    OPTIMIZER="SGD"
    METRICS=["accuracy"]

    new_model.compile(loss=LOSS_FUNCTION,optimizer=OPTIMIZER,metrics=METRICS)
    
    #Train the model
    EPOCS=10
    VALIDATION=(X_valid,y_valid_bin)
    Transfer_learning_model=new_model.fit(X_train,y_train_bin,epochs=EPOCS,validation_data=VALIDATION)
    path_to_dir=os.path.join("artifacts","saved_models")
    model_file_path=os.path.join(path_to_dir,"transfer_model.h5")
    new_model.save(model_file_path)
    #model evaluation and logging
    logging.info(f"base model is saved st : {model_file_path}")
    logging.info(f">>>>>>> Model evaluation : {new_model.evaluate(X_test,y_test_bin)}")



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    # args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        # params_path=parsed_args.params
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e