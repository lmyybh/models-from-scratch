# en-zh-translation 英译中模型

这是一个简单的英译中模型训练和推理的案例，目前支持 Transformer 模型。

数据集下载地址：https://www.heywhale.com/mw/dataset/60c41b7a19d601001898b34a/file

## 模型训练

修改 `config.yaml` 文件中的模型配置

```bash
python -m models-from-scratch.examples.en-zh-translation.train --config </path/to/config.yaml>
```

## 模型推理

```bash
python -m models-from-scratch.examples.en-zh-translation.inference --config </path/to/config.yaml> --ckpt </path/to/model.pth> --gpu <int>
```

