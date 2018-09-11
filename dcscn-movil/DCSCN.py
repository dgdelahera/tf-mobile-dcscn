"""
Paper: "Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network"
Ver: 2.0

DCSCN model implementation (Transposed-CNN / Pixel Shuffler version)
See Detail: https://github.com/jiny2001/dcscn-super-resolution/

Please note this model is updated version of the paper.
If you want to check original source code and results of the paper, please see https://github.com/jiny2001/dcscn-super-resolution/tree/ver1.
"""

import logging
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.image_ops_impl import ResizeMethod

from helper import tf_graph, utilty as util

BICUBIC_METHOD_STRING = "bicubic"


class SuperResolution(tf_graph.TensorflowGraph):
    def __init__(self, flags, model_name=""):

        super().__init__(flags)

        # Model Parameters
        self.scale = flags.scale
        self.layers = flags.layers
        self.filters = flags.filters
        self.min_filters = min(flags.filters, flags.min_filters)
        self.filters_decay_gamma = flags.filters_decay_gamma
        self.use_nin = flags.use_nin
        self.nin_filters = flags.nin_filters
        self.nin_filters2 = flags.nin_filters2
        self.reconstruct_layers = max(flags.reconstruct_layers, 1)
        self.reconstruct_filters = flags.reconstruct_filters
        self.resampling_method = BICUBIC_METHOD_STRING
        self.pixel_shuffler = flags.pixel_shuffler
        self.pixel_shuffler_filters = flags.pixel_shuffler_filters
        self.self_ensemble = flags.self_ensemble

        # Image Processing Parameters
        self.max_value = flags.max_value
        self.channels = flags.channels
        self.output_channels = 1
        self.psnr_calc_border_size = flags.psnr_calc_border_size
        if self.psnr_calc_border_size < 0:
            self.psnr_calc_border_size = 2 + self.scale

        # initialize variables
        self.name = self.get_model_name(model_name)
        self.total_epochs = 0

        util.make_dir(self.checkpoint_dir)
        logging.info("\nDCSCN v2-------------------------------------")
        logging.info("%s [%s]" % (util.get_now_date(), self.name))

    def get_model_name(self, model_name, name_postfix=""):
        if model_name is "":
            name = "dcscn_L%d_F%d" % (self.layers, self.filters)
            if self.min_filters != 0:
                name += "to%d" % self.min_filters
            if self.filters_decay_gamma != 1.5:
                name += "_G%2.2f" % self.filters_decay_gamma
            if self.cnn_size != 3:
                name += "_C%d" % self.cnn_size
            if self.scale != 2:
                name += "_Sc%d" % self.scale
            if self.use_nin:
                name += "_NIN"
                if self.nin_filters != 0:
                    name += "_A%d" % self.nin_filters
                if self.nin_filters2 != self.nin_filters // 2:
                    name += "_B%d" % self.nin_filters2
            if self.pixel_shuffler:
                name += "_PS"
            if self.max_value != 255.0:
                name += "_M%2.1f" % self.max_value
            if self.activator != "prelu":
                name += "_%s" % self.activator
            if self.batch_norm:
                name += "_BN"
            if self.reconstruct_layers >= 1:
                name += "_R%d" % self.reconstruct_layers
                if self.reconstruct_filters != 1:
                    name += "F%d" % self.reconstruct_filters
            if name_postfix is not "":
                name += "_" + name_postfix
        else:
            name = "dcscn_%s" % model_name

        return name

    def build_graph(self):

        self.input = tf.placeholder(tf.float32, shape=[1, 500, 500, 1], name="input")
        self.y = tf.placeholder(tf.float32, shape=[None, None, None, self.output_channels], name="y")

        if self.max_value != 255.0:
            self.input = np.multiply(self.input, self.max_value / 255.0)

        #bicubic_input_image = tf.image.resize_images(self.input, [1000,1000], method=ResizeMethod.BICUBIC)

        x = tf.reshape(self.input, [1, 500, 500, 1])
        #x2 = tf.reshape(bicubic_input_image, [1, 1000, 1000, 1])

        # building feature extraction layers

        output_feature_num = self.filters
        total_output_feature_num = 0
        input_feature_num = self.channels
        input_tensor = x


        for i in range(self.layers):
            if self.min_filters != 0 and i > 0:
                x1 = i / float(self.layers - 1)
                y1 = pow(x1, 1.0 / self.filters_decay_gamma)
                output_feature_num = int((self.filters - self.min_filters) * (1 - y1) + self.min_filters)

            self.build_conv("CNN%d" % (i + 1), input_tensor, self.cnn_size, input_feature_num,
                            output_feature_num, use_bias=True, activator=self.activator,
                            use_batch_norm=self.batch_norm, dropout_rate=1)
            input_feature_num = output_feature_num
            input_tensor = self.H[-1]
            total_output_feature_num += output_feature_num

        with tf.variable_scope("Concat"):
            self.H_concat = tf.concat(self.H, 3, name="H_concat")
        self.features += " Total: (%d)" % total_output_feature_num

        # building reconstruction layers ---

        if self.use_nin:
            self.build_conv("A1", self.H_concat, 1, total_output_feature_num, self.nin_filters,
                            dropout_rate=self.dropout_rate, use_bias=True, activator=self.activator)
            self.receptive_fields -= (self.cnn_size - 1)

            self.build_conv("B1", self.H_concat, 1, total_output_feature_num, self.nin_filters2,
                            dropout_rate=self.dropout_rate, use_bias=True, activator=self.activator)

            self.build_conv("B2", self.H[-1], 3, self.nin_filters2, self.nin_filters2,
                            dropout_rate=self.dropout_rate, use_bias=True, activator=self.activator)

            self.H.append(tf.concat([self.H[-1], self.H[-3]], 3, name="Concat2"))
            input_channels = self.nin_filters + self.nin_filters2
        else:
            self.H.append(self.H_concat)
            input_channels = total_output_feature_num

        # building upsampling layer
        if self.pixel_shuffler:
            if self.pixel_shuffler_filters != 0:
                output_channels = self.pixel_shuffler_filters
            else:
                output_channels = input_channels
            if self.scale == 4:
                self.build_pixel_shuffler_layer("Up-PS", self.H[-1], 2, input_channels, input_channels)
                self.build_pixel_shuffler_layer("Up-PS2", self.H[-1], 2, input_channels, output_channels)
            else:
                self.build_pixel_shuffler_layer("Up-PS", self.H[-1], self.scale, input_channels, output_channels)
            input_channels = output_channels
        else:
            self.build_transposed_conv("Up-TCNN", self.H[-1], self.scale, input_channels)

        for i in range(self.reconstruct_layers - 1):
            self.build_conv("R-CNN%d" % (i + 1), self.H[-1], self.cnn_size, input_channels, self.reconstruct_filters,
                            dropout_rate=self.dropout_rate, use_bias=True, activator=self.activator)
            input_channels = self.reconstruct_filters

        self.build_conv("R-CNN%d" % self.reconstruct_layers, self.H[-1], self.cnn_size, input_channels,
                        self.output_channels)

        #self.y_ = tf.add(self.H[-1], x2, name="output_tensor")
        self.y_ = tf.identity(self.H[-1], name="output_tensor")



        logging.info("Feature:%s Complexity:%s Receptive Fields:%d" % (
            self.features, "{:,}".format(self.complexity), self.receptive_fields))

    def do(self, input_image, bicubic_input_image=None):

        y = self.sess.run(self.y_, feed_dict={self.input: input_image.reshape(1, 500, 500, 1)})
        output = y[0]

        if self.max_value != 255.0:
            hr_image = np.multiply(output, 255.0 / self.max_value)
        else:
            hr_image = output

        return hr_image

    def do_for_file(self, file_path, output_folder="output"):
        org_image = util.load_image(file_path)

        filename, extension = os.path.splitext(os.path.basename(file_path))
        output_folder += "/" + self.name + "/"
        util.save_image(output_folder + filename + extension, org_image)

        #Esta linea para solo blanco y negro
        org_image = util.convert_rgb_to_y(org_image)
        util.save_image(output_folder + filename + "_y" + extension, org_image)

        if len(org_image.shape) >= 3 and org_image.shape[2] == 3 and self.channels == 1:
            input_y_image = util.convert_rgb_to_y(org_image)
            scaled_image = util.resize_image_by_pil(input_y_image, self.scale, resampling_method=self.resampling_method)
            util.save_image(output_folder + filename + "_bicubic_y" + extension, scaled_image)
            output_y_image = self.do(input_y_image)
            util.save_image(output_folder + filename + "_result_y" + extension, output_y_image)

            scaled_ycbcr_image = util.convert_rgb_to_ycbcr(
                util.resize_image_by_pil(org_image, self.scale, self.resampling_method))
            image = util.convert_y_and_cbcr_to_rgb(output_y_image, scaled_ycbcr_image[:, :, 1:3])

        else:
            scaled_image = util.resize_image_by_pil(org_image, self.scale, resampling_method=self.resampling_method)
            util.save_image(output_folder + filename + "_bicubic_y" + extension, scaled_image)
            image = self.do(org_image)
            util.save_image(output_folder + filename + "_residual_y" + extension, image)
            final_image = scaled_image + image



        util.save_image(output_folder + filename + "_result" + extension, final_image)

