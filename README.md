# slam #

This repository contains a slimmed down mono slam algorithm.

## Building ##

It should successfully build using the provided Dockerfile which will automatically install all dependencies.

The build configuration is handled by CMake.

Example build process using docker
```
bash setup.sh
mkdir build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
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
