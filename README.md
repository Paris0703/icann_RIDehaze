# icann_RIDehaze

This is the GitHub content for ICANN 2024, titled 'A robust image dehazing model using cycle generative adversarial network with an improved atmospheric scatter model.'
The paper has now been pre-accepted, and the test.py are now available. The complete files will be released after publication

**************2024-9-11**************

The paper has been accepted, and the conference will be held in Lugano, Switzerland, on September 17, 2024. The complete code will be released after the conference on September 20. Currently, all code except for `train.py` is available.
The `network` folder contains the network used to estimate image depth, originally available on GitHub at the following link: [https://github.com/nianticlabs/wavelet-monodepth](https://github.com/nianticlabs/wavelet-monodepth).

**************2024-9-24**************

The code has been fully open-sourced, including `train.py` and `.pth` files. Note that `train_fine_tuning.py` is for fine-tuning a model trained on a large dataset (such as RESIDE) on smaller datasets.
pth file:  https://drive.google.com/file/d/11O4JONP_ep1c06qANy0OliD7j9Acln05/view?usp=drive_link
