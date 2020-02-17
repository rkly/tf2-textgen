# tf2-textgen
Generate text using a character-based RNN
based on https://www.tensorflow.org/tutorials/text/text_generation

Used [Amazon Foode Review Dataset](https://www.kaggle.com/snap/amazon-fine-food-reviews)

get_reviews.py extracts text from csv

textgen_train.py starts training

textgen_predict.py rebuild model with input shape (1, None)

Due to this [bug](https://github.com/tensorflow/tensorflow/issues/34644)
is unable to save model in tf format for further conversion into tflite formar for serving.
