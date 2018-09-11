"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Author: Jin Yamanaka
Github: https://github.com/jiny2001/dcscn-image-super-resolution
Ver: 2.0

Apply Super Resolution for image file.

--file [your image filename]: will generat HR images.
see output/[model_name]/ for checking result images.

Also you must put same model args as you trained.
For ex, if you trained like
python3 train.py --layers 4 --filters 24 --dataset test --training_images 400

Then you must run evaluate.py like below.
python3 evaluate.py --layers 4 --filters 24 --file your_image_file_path
"""

import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod

import DCSCN
from helper import args
from helper import loader, utilty as util
import os


args.flags.DEFINE_string("file", "image.jpg", "Target filename")
FLAGS = args.get()


def main(_):
       model = DCSCN.SuperResolution(FLAGS, model_name=FLAGS.model_name)
       model.build_graph()


       model.init_all_variables()
       model.load_model()

       model.do_for_file(FLAGS.file, FLAGS.output_dir)

    # # Codigo para evaluación
    #  file_path='FotosDani/eval/'
    #  true_image = util.load_image('FotosDani/img_001_input.png')
    #  input_y_image = util.resize_image_by_pil(true_image, 2)
    #  # util.save_image('FotosDani/27.jpg', input_y_image)
    #
    #  for image in os.listdir(file_path):
    #      output_y_image = util.load_image(file_path + image)
    #      mse = util.compute_mse(input_y_image, output_y_image, border_size=-1)
    #      print("Imagen: ", image, "PSNR: ", util.get_psnr(mse))

     #  true_image = util.load_image('FotosDani/dragon_original.jpg')
     #  output_y_image = util.load_image('FotosDani/dragon_result_ordenador.jpg')
     # # #r, g, b = output_y_image[:, :, 0], output_y_image[:, :, 1], output_y_image[:, :, 2]
     # # #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
     #  input_y_image = util.resize_image_by_pil(true_image, 2)
     #  mse = util.compute_mse(input_y_image, output_y_image, border_size=-1)
     #  print("Imagen: Oso ", "PSNR: ", util.get_psnr(mse))


if __name__ == '__main__':
    tf.app.run()
