# This file is used to configure the training parameters for each task
import os.path


class Config_Cardiac_multi_plane:
    # This dataset contain all the collected ultrasound dataset
    data_path = "../data/US_Cardiac_multi_plane"
    save_path = "../save/US_Cardiac_multi_plane/"
    result_path = "../result/US_Cardiac_multi_plane"
    tensorboard_path = "/tensorboard/US_Cardiac_multi_plane"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 100                        # number of total epochs to run (default: 400)
    batch_size = 8                      # batch size (default: 4)
    learning_rate = 5e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    train_split = "train"               # the file name of training set
    val_split = "val"                   # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"            # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "train"
    visual = False
    modelname = "SAM"


class Config_Cardiac_multi_plane_test:
    # This dataset contain all the collected ultrasound dataset
    data_path = "../data/US_Cardiac_multi_plane"
    data_subpath = os.path.join(data_path, 'cluster1')
    result_path = "../result/US_Cardiac_multi_plane"
    load_path = "../save/.pth"
    save_path_code = "_"

    workers = 1                         # number of data loading workers (default: 8)
    epochs = 200                        # number of total epochs to run (default: 400)
    batch_size = 8                      # batch size (default: 4)
    learning_rate = 5e-4                # iniial learning rate (default: 0.001)
    momentum = 0.9                      # momntum
    classes = 2                         # thenumber of classes (background + foreground)
    img_size = 256                      # theinput size of model
    test_split = "test_cluster1"        # the file name of testing set
    crop = None                         # the cropped image size
    eval_freq = 1                       # the frequency of evaluate the model
    save_freq = 2000                    # the frequency of saving the model
    device = "cuda"                     # training device, cpu or cuda
    cuda = "on"                         # switch on/off cuda option (default: off)
    gray = "yes"                        # the type of input image
    img_channel = 1                     # the channel of input image
    eval_mode = "mask_slice"                 # the mode when evaluate the model, slice level or patient level
    pre_trained = False
    mode = "test"
    visual = False
    modelname = "SAM"


# ==================================================================================================
def get_config(task="US30K"):
    if task == "Cardiac_multi_plane":
        return Config_Cardiac_multi_plane()
    elif task == "Cardiac_multi_plane_test":
        return Config_Cardiac_multi_plane_test()
    else:
        assert("We do not have the related dataset, please choose another task.")
