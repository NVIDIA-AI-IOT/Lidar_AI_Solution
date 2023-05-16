# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import cuosd
import cv2
import time

class timer:
    def __init__(self, name):
        self.name = name

    def __call__(self, f):
        def counter(*args):
            begin = time.time()
            ret = f(*args)
            end   = time.time()
            print(f"{self.name} {(end - begin) * 1000:.3f} ms")
            return ret
        return counter

@timer("OpenCV")
def opencv_draw(image):
    box   = cv2.boxPoints(((256, 256), (350, 200), 30)).astype(int)
    for i in range(1000):
        cv2.putText(image, "test", (80, 200), 0, 4, (255, 0, 0), 5, 16)
        cv2.putText(image, "NVIDIA cuOSD", (80, 200), 0, 5, (255, 0, 0), 10, 16)
        cv2.rectangle(image, (100, 100), (800, 500), (0, 255, 0), -1)
        cv2.circle(image, (500, 500), 200, (255, 0, 255), -1)
        cv2.circle(image, (300, 200), 100, (0, 255, 255), -1)
        cv2.drawContours(image, [box], 0, (0, 0, 255), -1)

def cuosd_draw(image):

    stream = drv.Stream()
    width, height = image.shape[1], image.shape[0]
    device_mem = drv.mem_alloc(width * height * 3)
    drv.memcpy_htod_async(device_mem, image, stream)
    osd = cuosd.Context()

    def draw():
        osd.text("test", 35, "data/simhei.ttf", 80, 100, (255, 0, 0), (0, 0, 0, 0))
        osd.text("NVIDIA cuOSD", 62, "data/simhei.ttf", 150, 200, (255, 0, 0), (0, 0, 0, 0))
        osd.rectangle(100, 100, 800, 500, 5, (0, 255, 0), (0, 128, 255, 128))
        osd.point(300, 200, 100, (0, 255, 255, 255))
        osd.circle(500, 500, 200, 9, (255, 0, 255, 200), (255, 255, 0, 100))
        osd.rotationbox(300, 300, 200, 350, np.pi * 0.3, 10, (0, 255, 0, 180), True, (255, 128, 0, 100))
        osd.apply(device_mem, 0, width, width * 3, height, cuosd.ImageFormat.RGB, stream.handle, True, True)

    # Warmup
    for i in range(100):
        draw()
        
    drv.memcpy_dtoh_async(image, device_mem, stream)
    stream.synchronize()

    @timer("cuOSD")
    def loop():
        for i in range(1000):
            draw()
        drv.memcpy_dtoh_async(image, device_mem, stream)
        stream.synchronize()
        
    draw()
    stream.synchronize()

    loop()


def performance_test():
    image = cv2.imread("imgs/input.png")
    opencv_image = image.copy()
    opencv_draw(opencv_image)
    cv2.imwrite("imgs/opencv.png", opencv_image)

    cuosd_image = image.copy()
    cuosd_draw(cuosd_image)
    cv2.imwrite("imgs/cuosd.png", cuosd_image)

def simple_test():
    image = cv2.imread("imgs/input.png")

    osd = cuosd.Context()
    osd.text("test", 35, "data/simhei.ttf", 80, 100, (255, 0, 0), (0, 0, 0, 0))
    osd.text("NVIDIA cuOSD", 62, "data/simhei.ttf", 150, 200, (255, 0, 0), (0, 0, 0, 0))
    osd.rectangle(100, 100, 800, 500, 5, (0, 255, 0), (0, 128, 255, 128))
    osd.point(300, 200, 100, (0, 255, 255, 255))
    osd.circle(500, 500, 200, 9, (255, 0, 255, 200), (255, 255, 0, 100))
    osd.rotationbox(300, 300, 200, 350, np.pi * 0.3, 10, (0, 255, 0, 180), True, (255, 128, 0, 100))
    osd(image)

    print("Save result to imgs/output.png")
    cv2.imwrite("imgs/output.png", image)
    
def blur_test():
    image = cv2.imread("imgs/faces.jpg")
    boxes = [
        [820.1912307739258,119.93292999267578,912.2005653381348,237.0315284729004],
        [46.04017639160156,177.24126434326172,145.02646255493164,296.9175262451172],
        [649.4729461669922,107.13701248168945,739.6138496398926,221.88798141479492],
        [355.8001022338867,123.4946403503418,449.8617935180664,247.0775909423828],
        [239.8397445678711,117.28738403320312,330.16421127319336,232.78676223754883],
        [517.8735160827637,128.19419860839844,603.5273056030273,239.37591171264648],
    ]
    
    osd = cuosd.Context()
    for box in boxes:
        osd.boxblur(*list(map(int, box)), 9)
    osd(image)

    print("Save result to imgs/blur-faces.jpg")
    cv2.imwrite("imgs/blur-faces.jpg", image)

if __name__ == "__main__":
    performance_test()
    simple_test()
    blur_test()