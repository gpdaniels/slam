# slam #

This repository contains a slimmed down mono slam algorithm.

## Building ##

It should successfully build using the provided Dockerfile which will automatically install all dependencies.
(Note: The docker container is setup to do X11 forwarding)

The build configuration is handled by CMake.

Example build process using docker
```
bash setup.sh
> mkdir build
> cd build
> cmake -DCMAKE_BUILD_TYPE=Release ..
> cmake --build . --config Release
```

## Usage ##

After building in `./build/` it should be possible to run the system using these commands:

Sample data from https://vision.in.tum.de/data/datasets/rgbd-dataset/download
```
./slam "../videos/freiburgxyz_525.mp4" 525
```

Sample data from https://www.cvlibs.net/datasets/kitti/
```
./slam "../videos/kitti_984.mp4" 984
```

## Demos ##

### freiburgxyz sequence ###
[![Video of freiburgxyz sequence](https://img.youtube.com/vi/SndmqN-2VQQ/0.jpg)](https://www.youtube.com/watch?v=SndmqN-2VQQ)

### kitti sequence ###
[![Video of kitti sequence](https://img.youtube.com/vi/M6buBEKaoq4/0.jpg)](https://www.youtube.com/watch?v=M6buBEKaoq4)


