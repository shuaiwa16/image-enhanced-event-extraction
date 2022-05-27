# Introduce
The source code of the paper *image enhanced event detection in news articles* in AAAI2020.

For event extration, we propose a novel Dual Recurrent Multimodal Model, DRMM, to conduct deep interactions between images and sentences for modality features aggregation.

Our academic paper which describes DRMM in detail can be found here: https://tongmeihan1995.github.io/meihan.github.io/research/AAAI2020.pdf.

### Quick Start

1. Get ace2005 English dataset and transform it to BIO format in EEdata directory, named train.txt, dev.txt and test.txt, and sentences are splitted by an empty line.

2. Download our image dataset in EEdata directory.

3. Put all files in EEdata directory.

4. Run sh run.sh, and set train = True if you want to retrain the model.


# How do I cite?
```
@inproceedings{tong2020image,
  title={Image enhanced event detection in news articles},
  author={Tong, Meihan and Wang, Shuai and Cao, Yixin and Xu, Bin and Li, Juanzi and Hou, Lei and Chua, Tat-Seng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={34},
  number={05},
  pages={9040--9047},
  year={2020}
}
```
