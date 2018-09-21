
# nmf

Assignment1 Non-negative matrix factorization

Author
------
Chen Chen 480458339

Xiaodan Gan 440581983

Xinyue Wang 440359463


Running Environment Setup
------------

We implemented NMF as a python package with modules of noise , file IO, evaluation metrics and visualization. Please follow the instruction to make sure the code will run properly.

1. Copy ORL and CroppedYaleB dataset into data folder. Please make sure you have the following file tree structure:
```
  |--- data
  	|--- ORL
  	|--- CroppedYaleB
  |--- code
      |--- nmf
        |--- __version__.py
        |--- algorithm.py
        |--- io.py
        |--- main.py
        |--- metric.py
        |--- noise.py
        |--- util.py
        |--- vis.py
      |--- setup.py
      |--- README.md
```
2. Navigate into folder code.

3. Install `nmf` with following command: (Please use `pip3` if the default `python` in your computer is `python2`)

   ```
   $ pip install -e .
   ```
 This command will run  `setup.py` where we specify the dependencies required to run  `nmf`. The dependencies we require are:

           "numpy>=1.14.0",
           "scipy>=1.0.0",
           "scikit-learn>=0.19.1",
           "pandas>=0.20.2",
           "matplotlib>=2.0.2",
           "Pillow>=5.2.0",
           "tqdm>=4.26.0",
Please note that if the version number of installed package in your machine is lower than the stated version number, `pip` will uninstall your out-of-date package and install the one with version number greater than or equal to the stated one in `setup.py`.

4. Run `main.py` in `nmf`.

   To run `nmf` on **ORL** dataset, please run:

   ```
   python nmf/main.py orl
   ```

   To run `nmf` on **CroppedYaleB** dataset, please run:

   ```
   python nmf/main.py croppedYale
   ```
   If no command line argument found, `main.py` will run ORL dataset by default.

Progress bar will show up to indicate the rough running time for each algorithm and noise combination. All results will be auto-saved to folder `results/{generated-time-dataname}`.

Note that we set the epoch to be 1 in `main.py`. This is because we have 4 (noise) x 2 (algorithm) = 8 combination in each epoch. However, we increased the epochs to calculate average metrics and confidence interval etc.

Also note that we have a flag `multi_start_flag` in `main.py` at line 17. If the flag is turned on (value set to 1), we will use multi-start parallel computing version of KLNMF to avoid getting stuck in local optimal. Currently we turn off the flag so that one epoch will finish in 1 minute. Feel free to turn it on to play around with code.
