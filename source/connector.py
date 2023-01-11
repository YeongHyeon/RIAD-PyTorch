def connect(nn):

    if(nn == 0): return 0

    elif(nn == 1000): import neuralnet.ae_l2 as nn # ae

    elif(nn == 2000): import neuralnet.aesc_l2 as nn # unet

    return nn
