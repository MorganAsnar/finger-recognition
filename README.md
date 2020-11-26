# finger-recognition
A software to recognize fingers

The proof of concept directory contains the software to test a full database, and will give the datas that can be found on the paper pdf paper included.

The usable prototype contains a software that will create a database from reference pictures, and one that load this database and gives you, for a picture, its class according to the algorithm of the proof of concept.

Please be aware that at the time of this writing, openCV package does not include SIFT algorithm. This should come soon, as the patent expired in march, but for now, you should import the contrib package, using the following: pip install opencv-contrib-python
