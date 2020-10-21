from threading import Thread
import cv2
from queue import Queue
import logging
import sys



#class to read streams using threads
class VideoThread():

    def __init__(self,path,logger,fps_rate=1,queue_size=100):
        self.path = path
        self.capture = cv2.VideoCapture(path)
        self.stopped = False
        self.queue   = Queue(maxsize=queue_size)
        self.logger  = logger
        #Increase to have more fps. Division of the original framerate. Integer.
        self.frame_put_rate = self.capture.get(cv2.CAP_PROP_FPS) // fps_rate
        self.logger.info('FRAME PUT RATE: {}/s'.format(fps_rate))
        self.logger.info('QUEUE SIZE: {}'.format(queue_size))
        self.fps_rate = fps_rate
        self.counter_frames = 0

    #start a thread that calls update method
    def start(self):
        thread = Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()
        self.logger.info('Video capturing thread has started')
        return

    def update(self):
        
        while True:

            if self.stopped:
                return
                
            if not self.queue.full():
                try:
                    #read current frame from capture
                    (ret,frame) = self.capture.read()
                    self.logger.debug('Read Capture: ret = {}'.format(ret))
                except Exception as e:
                    #restart 
                    self.logger.error('Exception captured: {}'.format(e))
                    self.logger.error('Capture released and reopened')
                    #if we can get a frame, the stream has finished
                if not ret:
                    self.logger.warn('Thread has stopped due to not ret')
                    self.capture.release()
                    self.capture.open(self.path)
                #add frame to queue
                self.counter_frames += 1
                
                if self.counter_frames % self.frame_put_rate == 0:
                    self.logger.debug('Added frame to queue')
                    self.queue.put(frame)
    
            #If the queue is full, clear it and reset
            else:
                self.queue.queue.clear()
                self.logger.info('Queue has been cleared as it has reached maximum capacity')
        self.logger.error('Out of while loop in video thread')
    
    def read(self): 
        #return next frame in queue
        return self.queue.get()

    def more(self):
        #return True if there are still frames in queue
        return self.queue.qsize() > 0

    def stop(self):
        #indicate that the thread should be stopped
        self.stopped = True


