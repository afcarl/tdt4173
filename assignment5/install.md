## Setup (Windows)

You need python 2.7 with pip, so make sure you have that

* Go to http://www.lfd.uci.edu/~gohlke/pythonlibs/
    * Download Pillow-3.2.0-cp27-cp27m-win32.whl
    * Download scikit_learn-0.17.1-cp27-cp27m-win32.whl
    * Download scikit_image-0.12.3-cp27-cp27m-win32.whl
    * Also download other needed libs, such as numpy, scipy, h5py
* You may need to install cython for h5py to work
* For each downloaded wheel binary, run `pip install that_filename.whl` (replace "that_filename.whl")

## Setup (Linux)

You need python 2.7 with pip, so make sure you have that

In theory, this command should install the needed dependencies:

`sudo pip install -r requirements.txt`

If a problem occurs, just google it
