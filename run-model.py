from model import model_
import os
import tensorflow as tf

if __name__=='__main__':

    get_m = model_.Model()
    get_m.model_run(status='predict', predict_w=' What are you? ')
    get_m.model_run(status='predict', predict_w=' I want to kill him ')

    

    