# Mu-BER-API
This repo includes Matlab and Python scripts required for BER calculation.
1. Use HDF5 with the original format to save the V^ matrix. 
    E.g. for RX1: /MuMIMO/6x6/RX1/VMat/real and /MuMIMO/6x6/RX1/VMat/image
2. Name the file 'MuMIMODataN.h5'. Note that old data files cannot be used, since calculating BER requires channel seeds.
3. Place the file in same directory as the Matlab and Python scripts are located.
4. Start a Matlab session, convert it to ashared session by matlab.engine.shareEngine
5. Run the APIMIMO.py file. This file extract V matrices for each user and send it to Matlab (BERcal.m) to calculate the BER. 
6. the out put of APIMIMO.py is an numpy array "BerMat" with dimenssion numberOfUser-by-umberOfSamples.

Output example for 20 samples:

BerMat

array([[0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.02115741, 0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ]])


BerMat.shape

(3, 20)

Notes:

* Ocasionally you may get this error, EngineError: Unable to connect to MATLAB session 'MATLAB_XXXXXX'.
Seems like this bug is not fixed yet; this is what helped me:

Start with removing all your variables and try again.
If clearing cache didn't help, restart the python/Matlab.

* File 'MuMIMODataN.h5' can be used to test the Python-Matlab API.
