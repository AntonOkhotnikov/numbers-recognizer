## Numbers audio recognizer

### Solution description

* No augmentations are applied, because I assume that test set contains similar clear TTS recordings (no speed/volume perturbations, no noises)

* Since we have a word "тысяча" almost in the middle of each sentence (for every 4-digits number and bigger), I insert a "*" symbol in target labels as an additional token (at position -4) that emulates the word "тысяча" with a hope that this will improve the quality somehow. The parameter `enrich_target (bool)` controls if this token is used. Example: `4892 -> 4*892`

* Since we try to recognize numbers itself (not on a character level), I tried to use wide convolutional kernels

* I use standard 13-dimensional MFCC as input features (with a frame size of 25 ms and shift if 10 ms)

* Each signal is padded with zeros to the length of the longest audio (which is ~3.9s)

* The NN architecture is a small copy of DeepSpeech with <PLACEHOLDER> N parameters. The idea is following:
```
    Audio -> MFCC -> CNN-feature-extractor -> RNN layers -> Greedy decoder -> CTC-loss
```


### TO-DO:
* Same gender batching
* Beam search decoder - https://github.com/parlance/ctcdecode


### Build
Build section with a Docker image building will be updated later


### Training

* Install requirements first
```bash
    python3 -m pip install -r requirements.txt
```

* Go to `src` folder and begin training with
```
    python3 training_pipeline.py [-h] -e <EXP_NAME> -d <DATA_ROOT> [-b <BATCH_SIZE>] [-n <NUM_EPOCHS>]
```

* See training logs with **Tensorboard**:
```bash
    tensorboard --logdir exps/<EXP_NAME>/logs --port 8008
```

### Inference
To be updated

### Resources
* https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch
* https://github.com/SeanNaren/deepspeech.pytorch
