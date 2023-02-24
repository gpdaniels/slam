docker run \
    --rm \
    --mount type=bind,source=$(pwd),target=/slam \
    --network=host \
    --env DISPLAY=${DISPLAY} \
    --env LIBGL_DRI3_DISABLE=1 \
    --mount type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix \
    --mount type=bind,source=${XAUTHORITY},target=/root/.Xauthority \
    --device=/dev/dri:/dev/dri \
    --device=/dev/video0:/dev/video0 \
    --interactive \
    --tty \
    $(docker build -q .)
