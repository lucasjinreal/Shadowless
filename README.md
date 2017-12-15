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

## Run
To run *Shadowless* after you get all pre-requirements, you can simply do:

```
python3 main.py

```
this will start a *Shadowless* master process to serve camera inputs and do the perception jobs.

## Detection

we are now deploy SSD and mask-rcnn and yolo-v2 on detection, it trains on a massive of coco image sets. It can
detect on more than 90 object, and maybe even more in the future.


## Segmentation

we now do segmentation based on mask-rcnn, this got a very good result now.

## Lane Detect

we now deploy lane detect in opencv methods.


# Copyright

this work inspired by Jin Fagang, you should not spread this soft-ware witout any guarantee, please using this under Apache License.
