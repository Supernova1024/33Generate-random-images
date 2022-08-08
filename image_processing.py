#!/usr/bin/python
# from pyimagesearch.shapedetector import ShapeDetector
import cv2, os
import math
import numpy as np
from scipy import stats
import imageio
from resizeimage import resizeimage
from skimage import filters
import cv2 as cv
import time
from numpy import array
import threading
import time
import random

input_folder = "All Surgery photos/"
# output_folder = "output/"
output_folder = "z_out1/"


def caculate_time_difference(start_milliseconds, end_milliseconds, filename):
   if filename == 'total':
      diff_milliseconds = int(end_milliseconds) - int(start_milliseconds)
      seconds=(diff_milliseconds / 1000) % 60
      minutes=(diff_milliseconds/(1000*60))%60
      hours=(diff_milliseconds/(1000*60*60))%24
      # print("Total run time", hours,":",minutes,":",seconds)
   else:
      diff_milliseconds = int(end_milliseconds) - int(start_milliseconds)
      seconds=(diff_milliseconds / 1000) % 60
      # print(seconds, "s", filename)

def img_mirror(img):
   img_mirror = cv2.flip(img, 1)
   return img_mirror

def img_rotate(img):
   ran = random.randint(1, 3)
   if ran == 1:
      image = cv.rotate(img, cv.ROTATE_180)
   if ran == 2:
      image = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
   if ran == 3:
      image = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
   return image

def img_blur(img):
   random_odd_numbers = random.randrange(5, 23+1, 8)
   blur_image = cv2.GaussianBlur(img, (7,7), 0)
   return blur_image

def img_resize(img):
   w = random.randint(440, 690)
   h = random.randint(440, 690)
   h, w = img.shape[:2]
   img = cv2.resize(img, ((w/3), (h/3)))
   return img

def img_bright(img):
   hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   flag = random.randint(0, 2) #whatever value you want to add
   bright1 = random.randint(0, 10) #whatever value you want to add
   bright2 = random.randint(0, 10) #whatever value you want to add
   bright3 = random.randint(0, 10) #whatever value you want to add
   if flag == 0:
      cv2.add(hsv[:,:,2], bright1, hsv[:,:,2])
   if flag == 1:
      cv2.add(bright2, hsv[:,:,2], hsv[:,:,2])
   if flag == 2:
      cv2.add(hsv[:,:,2], hsv[:,:,2], bright3)
   img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
   return img

def increase_brightness(img):
   value = random.randint(5, 200)
   hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
   h, s, v = cv2.split(hsv)

   lim = 255 - value
   v[v > lim] = 255
   v[v <= lim] += value

   final_hsv = cv2.merge((h, s, v))
   img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
   return img

def img_opacity(img):
   value1 = random.randint(235, 250)
   value2 = random.randint(1, 5)
   img = np.array(img, dtype=np.float)
   img /= value1
   a_channel = np.ones(img.shape, dtype=np.float)/value2
   image = img*a_channel
   return image

def img_crop(img):
   flag = random.randint(0, 3)
   h, w = img.shape[:2]
   x = random.randint(0, 50)
   y = random.randint(0, 50)
   if flag == 0:
      crop_img = img[y:h, x:w]
   if flag == 1:
      crop_img = img[0:h-y, x:w]
   if flag == 2:
      crop_img = img[y:h, 0:w-x]
   if flag == 3:
      crop_img = img[y:h-y, x:w-x]

   return crop_img


def processing_fun(img):
   for i in range(1, 20):
      flag1 = random.randint(0, 1)
      flag2 = random.randint(0, 3)
      flag3 = random.randint(0, 1)
      flag4 = random.randint(0, 1)
      flag5 = random.randint(0, 1)
      flag6 = random.randint(0, 5)

      

      if flag1 == 1:
         img = img_mirror(img)

      img1 = img_rotate(img)
      
      if flag2 == 1:
         img1 = img_blur(img1)

      if flag3 == 1:
         img1 = increase_brightness(img1)

      # img = img_resize(img)
      if flag5 == 1:
         img1 = img_crop(img1)
      # if flag6 != 1:
      # im_rgb = img[...,::-1]
      # if flag4 == 1:
      # img = img_opacity(img)
      im_rgb = cv2.cvtColor(img1, cv.COLOR_BGR2RGB)
      time_str = str(int(round(time.time() * 1000)))
      filename = "output" + time_str + ".jpg"
      imageio.imwrite(output_folder + filename, im_rgb)
      print("**************filename************", filename)

global thread_kill_flags

def processing(threadID, files):
   for filename in files:
      start_milliseconds = str(int(round(time.time() * 1000)))
      Cimg = cv2.imread(input_folder + filename)
      if Cimg is not None:
         processing_fun(Cimg);
      end_milliseconds = str(int(round(time.time() * 1000)))
      caculate_time_difference(start_milliseconds, end_milliseconds, filename)

class myThread (threading.Thread):
   def __init__(self, threadID, name, files, start_time):
      threading.Thread.__init__(self)
      self._stop = threading.Event()
      self.threadID = threadID
      self.name = name
      self.files = files
      self.start_time = start_time
   def stop(self):
      self._stop.set()

   def run(self):
      # print("====================", self.name)
      processing(self.threadID, self.files,)
      self.stop()
      end = str(int(round(time.time() * 1000)))
      caculate_time_difference(self.start_time, end, 'total')

def main(folder, start_time):
   stop_threads = False 
   filenames = []
   thread_list = []
   count_thread = 10
   
   for filename in os.listdir(folder):
      filenames.append(filename)

   mode = len(filenames) % (count_thread - 1)
   step = len(filenames) / (count_thread - 1)
   if len(filenames) < 20:
      start_total = str(int(round(time.time() * 1000)))
      for filename in os.listdir(folder):
         start_milliseconds = str(int(round(time.time() * 1000)))
         Cimg = cv2.imread(os.path.join(folder, filename))
         if Cimg is not None:
            processing_fun(Cimg);
         end_milliseconds = str(int(round(time.time() * 1000)))
         caculate_time_difference(start_milliseconds, end_milliseconds, filename)
      end_total = str(int(round(time.time() * 1000)))
      caculate_time_difference(start_total, end_total, "total")
   else:
      for i in range(1, count_thread):
         files = filenames[int(step)*(i-1) : int(step)*i]
         # Create new threads
         thread = myThread(i, "Thread_"+str(i), files, start_time)
         thread_list.append(thread)

      # Start new Threads
      for thread in thread_list:
         thread.start()

      if mode != 0:
         # Start mode Threads
         files = filenames[(count_thread - 1)*int(step):]
         thread1 = myThread(count_thread, "Thread_"+str(count_thread), files, start_time)
         thread1.start()

   
if __name__ == '__main__':
   start_time = str(int(round(time.time() * 1000)))
   
   main(input_folder, start_time)
   # main1()
