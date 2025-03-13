from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from os.path import join
from numpy import argmax
import re
import time

# 
feature_extractor_path = join( "model", "CNN_LSTM_best_model", "feature_extractor.h5" )
tokenizer_path = join( "model", "CNN_LSTM_best_model", "tokenizer.json")
lstm_model_path = join( "model", "CNN_LSTM_best_model", "CNN_LSTM_best_model.keras" )

def clean_sentence( sentence ):
  sentence = re.sub(r'<[^>]+>', '', sentence)

  sentence = sentence.replace( '_', ' ' )

  sentence = sentence.strip()  # Xóa khoảng trắng thừa
  sentence = " ".join( sentence.split() ) # Xóa khoảng trắng
  return str( sentence[0].upper() + sentence[1:] + '.' )  # viết hoa chữ cái đầu tiên, thêm dấu '.'
# end

def cnn_lstm_generate( image_path ):
    featurer_extractor = load_model( feature_extractor_path )
    model = load_model( lstm_model_path )

    with open( tokenizer_path ) as f:
        data = f.read()
        tokenizer = tokenizer_from_json( data )
        f.close()

    # extract image feature
    img = load_img( image_path, target_size=(299, 299) )
    img = img_to_array( img )
    img = img.reshape( (1, img.shape[0], img.shape[1], img.shape[2]) )

    # Apply InceptionV3 preprocessing
    img = preprocess_input( img )

    # Extract features
    feature = featurer_extractor.predict( img, verbose=0 )

    # generate
    seed_text = '<start>'

    while True:
        seed_seq = tokenizer.texts_to_sequences( [ seed_text ] )[0]
        seed_seq = pad_sequences( [seed_seq], maxlen = 40, padding = 'pre' )

        y_pred = model.predict( [ feature, seed_seq ], verbose = 0 )
        y_pred = argmax( y_pred )

        word = tokenizer.index_word[ y_pred ]
        seed_text += ' ' + word

        if word == '<end>':
            break
  #  end while

    return clean_sentence( seed_text )
# end def


if __name__ == "__main__":
    start_time = time.time()

    print( cnn_lstm_generate( join( "static", "image", "tuyenvn_20210620094210.jpg" ) ) )
    
    print( time.time() - start_time )

    # model = load_model( lstm_model_path )
    # print( model.summary() )