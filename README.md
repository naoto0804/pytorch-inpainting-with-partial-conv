# pytorch-inpainting-with-partial-conv

**Note that this is an ongoing project and I cannot fully reproduce the results. Suggestions and PRs are welcome!**

This is an unofficial pytorch implementation of a paper, [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723) [Liu+, arXiv2018].

## Requirements
- Python 3.6+
- Pytorch 0.4.1+

```
pip install -r requirements.txt
```

## Usage

### Preprocesse 
Generate masks by following [1] (saved under `./masks` by default).

**Note that the way of the mask generation is different from the original work**

```
python generate_data.py
```

### Train
```
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py
```

### Fine-tune
```
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py --finetune --resume <checkpoint_name>
```

## Results

Here are some results from the test set after the training of 500,000 iterations and fine-tuning (freezing BN in encoder) of 500,000 iterations.
![Results](result_iter_1000000.png)

## TODO
- [] Check the quality

## References
- [1]: [Unofficial implementation in Chainer](https://github.com/SeitaroShinagawa/chainer-partial_convolution_image_inpainting)
