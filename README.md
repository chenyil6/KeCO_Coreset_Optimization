# Enhancing Multimodal In-Context Learning for Image Classification through Coreset Optimization

This is the official project repository for our paper *Enhancing Multimodal In-Context Learning for Image Classification through Coreset Optimization.*

# Quick start

### Environment
Our installation follow the installation of [OpenFlamingo](https://github.com/mlfoundations/open_flamingo)

- ​For OpenFlamingo evaluation, use transformers version 4.33.2.
- For IDEFICS and Qwen2-VL evaluation, upgrade transformers to a newer version (e.g., 4.51.3).
- For Qwen2-VL runtime issues, you can refer to the original [Qwen2-VL](https://github.com/QwenLM/Qwen2.5-VL) repository.

### Get the results on CUB-200

```
cd CUB_200
python main.py --device {device} --model {model_name} --method {method}
# method should be in: (FewShot, Offline_ICL, Online_ICL).
```

### Get the results on ImageNet-100
```
cd ImageNet
python main.py --device {device} --model {model_name} --method {method}
# method should be in: (FewShot, Offline_ICL, Online_ICL).
```

### Get the results on Stanford Dogs
```
cd Stanford_dogs
python main.py --device {device} --model {model_name} --method {method}
# method should be in: (FewShot, Offline_ICL, Online_ICL).
```