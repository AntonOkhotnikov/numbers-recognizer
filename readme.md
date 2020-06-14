## 


### Description
Architecture and so on goes here


* No augmentations are applied, because I assume that test set contains similar clear TTS recordings

* Since we have a word "тысяча" almost in the middle of each sentence, I would insert a "_" symbol as a token that emulates the word "тысяча". I hope this will improve the quality somehow

* Still requires log-mel-filterbanks (MFCC) features: CNN-based encoder with a following recurrent (LSTM) decoder.

* I assume to minimize the CTC-loss when passing the CNN-extracted features from signal

* Probably need to combine Cross-Entropy loss for symbols classified. And then combine CE-loss with CTC-loss when passing all the features to recurrent decoder


### TO-DO:
* Dataloaders - returns [len, 13] array, where `len` varies depending on real length. Also returns `target` sequence
* Loss combine
* Tensorboard metrics callbacks
* Fix net input? by padding maybe
* Currently - random batch generation
* Max audio len is 3.831500
* Define big CNN kernel size and stride - because want to detect words, not symbols
* Beam search decoder? - https://github.com/parlance/ctcdecode


### Build
Please go to `build` and build docker image first


### Training


### Inference



### Possible improvements
* Since input size is fixed - try attention instead of BiLSTM?

### Resources
* https://www.assemblyai.com/blog/end-to-end-speech-recognition-pytorch
