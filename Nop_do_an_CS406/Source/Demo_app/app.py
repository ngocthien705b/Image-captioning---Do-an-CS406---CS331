from flask import Flask, render_template, request, flash, redirect
from cnn_lstm_generator import cnn_lstm_generate
from transformer_generator import vit_gpt2_generate
from os.path import join
from time import time


UPLOAD_FOLDER = join( 'static', 'uploads' )

app = Flask(__name__)
app.secret_key = "DNkir89HjuA"

app.config[ 'UPLOAD_FOLDER' ] = UPLOAD_FOLDER
app.config[ 'ALLOWED_EXTENSIONS' ] = { 'png', 'jpg', 'jpeg', 'gif' }

def allowed_file( filename ):
    return '.' in filename and filename.rsplit( '.', 1 )[1].lower() in app.config['ALLOWED_EXTENSIONS']
# end def

@app.route( '/' )
def index():
    return render_template( 'index.html' )
# end

@app.route( "/", methods=['POST'] )
def get_image():
    if "image" not in request.files:
        flash( "No image found" )
        return redirect(request.url)
    # end if
    
    image = request.files[ "image" ]
    if image.filename == "":
        flash( "No image selected" )
        return redirect( request.url )
    # end if

    if image and allowed_file( image.filename ):
        # save uploaded image
        file_tail = image.filename.rsplit( '.', 1 )[1].lower()
        file_name = "upload.{}".format( file_tail )
        save_path = join( app.config['UPLOAD_FOLDER'], file_name )
        image.save( save_path )

        redirect( request.url )

        # generate_caption
        start = time()
        cnn_lstm_caption = cnn_lstm_generate( save_path )
        cnn_lstm_time = "{:.2f}".format(float( time() - start ))

        start = time()
        vit_gpt2_caption = vit_gpt2_generate( save_path )
        vit_gpt2_time = "{:.2f}".format(float( time() - start ))

        return render_template('index.html',
                               image_path = save_path,
                               cnn_lstm_caption = cnn_lstm_caption,
                               cnn_lstm_time = cnn_lstm_time,
                               vit_gpt2_caption = vit_gpt2_caption,
                               vit_gpt2_time = vit_gpt2_time )
    else:
        flash( 'Allowed image types are - png, jpg, jpeg, gif' )
        return redirect( request.url )
    # end if else
# end def

if __name__ == "__main__":
    app.run( debug=False )