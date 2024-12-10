本项目根据torch的官方，mask-rcnn进行修改，可以训练并预测自己的数据集，数据集格式是coco的格式，虽说是coco格式，但是不同软件导出来的不一样，我用的是https://app.roboflow.com/，这个网站上面在线标注，然后导出coco数据集格式，这个网站还是非常方便的推荐大家使用，里面还有很多已经标注好的公开数据集。

有需要可以自行修改Datasets类即可，保证返回到对象是一致的即可

**用法**

```bash
#训练
python main.py --dataset_path=数据集的路径名称 --mode=train

#预测
python main.py --dataset_path=数据集的路径名称 --mode=predict --model_path=训练好模型的路径 --threshold=预测置信度，默认是0.7
```

