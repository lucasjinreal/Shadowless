# Shadowless

> The more fast your are, the more shadowless you got.


![](http://www.z4a.net/images/2017/11/21/frame_0019.jpg)

![](https://i.loli.net/2017/12/15/5a3338d6c3527.jpg)
Shadowless is a new generation auto-drive perception system that feel things only in vision(more features maybe add in).
We building shadowless on the purpose of establish a fully intelligent and fast drive system.

The 3 main part of shadowless are:

- **Detection**: this part implements many state-of-art detection algorithms to detect objects;
- **Segmentation**: currently we seg only on objects;
- **Lane Detect**: this will tell vehicle where should to ride and avoid touch the lane in 2 sides.


## Preparation
You need to do some setups to run *Shadowless*, you should download mxnet_ssd model from mxnet official examples repo.
and you should download one video from Youtube or anywhere place it into: `videos/` directory. then:

```
sudo pip3 install -r requirements.txt
```

Please do 2 steps before you start running `python3 main.py`:

- Download the pretrained model from [here](https://github.com/zhreshold/mxnet-ssd), currently you should download `Resnet-50 512x512`, directly download url is [here](https://github.com/zhreshold/mxnet-ssd/releases/download/v0.7-alpha/ssd_resnet50_512_voc0712trainval.zip), after downloaded untar it into mxnet_ssd/model dir;
- You should install mxnet with the newest version.
- Mask-RCNN backend for detection will release very soon



## Run

To run *Shadowless* after you get all pre-requirements, you can simply do:

```
python3 main.py

```
this will start a *Shadowless* master process to serve camera inputs and do the perception jobs.



## Contribute

So much welcome the community contribute your code to *Shadowless*, we are now need those features:

- [x] Detection with SSD
- [x] Detection with FasterRCNN
- [ ] Detection with RFCN
- [x] Lane Segment using OpenCV
- [ ] Lane Segment with DeepLearning methods
- [ ] Distance estimate with Objects


- [ ] Speed estimate with moving objects


- [ ] Accelerate whole networks to a real-time speed.



##Copyright

this work inspired by Jin Fagang, you should not spread this soft-ware witout any guarantee, please using this under Apache License.
