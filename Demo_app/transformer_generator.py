from transformers import TFVisionEncoderDecoderModel, ViTFeatureExtractor, GPT2Tokenizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from os.path import join
import time

model_dir = join( "model", "ViT_GPT2_Vn_model" )
model_path = join( model_dir, "checkpoint-2886" )
tokenizer_path = join( model_dir, "tokenizer" )
feature_extractor_path = join( model_dir, "feature_extractor" )

def vit_gpt2_generate( image_path ):

    # load model
    model = TFVisionEncoderDecoderModel.from_pretrained( model_path )
    feature_extractor = ViTFeatureExtractor.from_pretrained( feature_extractor_path )
    tokenizer = GPT2Tokenizer.from_pretrained( tokenizer_path )

    # load image
    image = load_img( image_path, target_size=(224, 224) )
    image = img_to_array( image )

    pixel_values = feature_extractor( images=image, return_tensors="tf" ).pixel_values

    generated_ids = model.generate( pixel_values, max_length=50 )
    generated_text = tokenizer.batch_decode( generated_ids, skip_special_tokens=True )[0]

    return generated_text.replace( ' .', '.' )
# end def

if __name__ == "__main__":
    start_time = time.time()

    print( vit_gpt2_generate( join( "static", "image", "tuyenvn_20210620094210.jpg" ) ) )

    print( time.time() - start_time )

    # model = TFVisionEncoderDecoderModel.from_pretrained( model_path )
    # print( model.summary() )