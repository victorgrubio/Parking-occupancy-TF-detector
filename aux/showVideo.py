# -*- coding: utf-8 -*-

import argparse
import cv2
import subprocess
import time

def launch_stream_ffmpeg(host,port):
    width = 640
    height = 480
    fps = 30
    stream_address = 'test.mp4'
    command_split  =  ['ffmpeg',
                      '-f', 'rawvideo',
                      '-s', '%dx%d'%(width, height),
                      '-pix_fmt', 'bgr24',
                      '-re',
                      '-i', '-',
                      '-an',
                      '-b:v', '512k',
                      '-bf', '0',
                      '-vcodec','mpeg4',
                      '-tune','zerolatency',
                      '-pix_fmt', 'yuv420p',
                      '-preset','ultrafast',
                       '-y',
                      stream_address]
    
    video_stream = subprocess.Popen(command_split,stdin=subprocess.PIPE,stderr=subprocess.PIPE,stdout=subprocess.PIPE)
    return video_stream

def launch_stream_hls(host,port):
    width = 640
    height = 480
    stream_address = 'www/videos/index.m3u8'
    command_split  =  ['ffmpeg',
                      '-f', 'rawvideo',
                      '-s', '%dx%d'%(width, height),
                      '-pix_fmt', 'bgr24',
                      '-re',
                      '-i', '-',
                      '-an',
                      '-f','hls',
                      '-hls_time','3',
                      '-g','3',
                      '-hls_list_size','3',
                      '-hls_wrap','3',
                      '-hls_segment_filename','www/videos/hls_stream_%03d.ts',
                      '-hls_flags','-delete_segments',
                      '-tune','zerolatency',
                      '-preset','ultrafast',
                       '-y',
                      stream_address]
    
    video_stream = subprocess.Popen(command_split,stdin=subprocess.PIPE,stderr=subprocess.PIPE,stdout=subprocess.PIPE)
    return video_stream


def launch_stream_vlc(host,port):
    command = 'cvlc --demux=rawvideo --rawvid-fps=25 --rawvid-width=640 --rawvid-height=480 - --sout "#transcode{vcodec=h264,vb=512,fps=25,width=640,height=480}:rtp{dst=localhost,port=8090,sdp=rtsp://localhost:8090/test.sdp}'
    command = "cvlc v4l2:///dev/video0 --sout '#transcode{vcodec=h264,vb=512}:standard{access=http,mux=ts,dst=:8081}'"
    command = 'cvlc --demux=rawvideo --rawvid-fps=30 --rawvid-width=640 --rawvid-height=480 --rawvid-chroma=RV24 - --sout "#transcode{vcodec=h264,vb=512,fps=30,width=640,height=480}:standard{access=http{mime=video/x-flv},mux=ffmpeg{mux=flv},dst=:8090/stream.flv}"' 
    command = 'cvlc --demux=rawvideo --rawvid-fps=30 --rawvid-width=640 --rawvid-height=480 --rawvid-chroma=RV24 - --sout "#transcode{vcodec=h264,vb=512,fps=30,width=640,height=480}:standard{access=http,mux=ts,dst=:8090}"'
    command_split = command.split(' ')
    video_stream = subprocess.Popen(command_split,stdin=subprocess.PIPE,stderr=subprocess.PIPE,stdout=subprocess.PIPE)
    return video_stream



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-v','--video',help='Video Path to stream',required=True)
    parser.add_argument('--stream',help='Stream video to port',action='store_true')
    parser.add_argument('--host',help='Host of stream',type=str,default='localhost')
    parser.add_argument('--port',help='Port of stream',type=str,default='8090')
    parser.add_argument('--display',help='Show CV window',action='store_true')
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(int(args.video))
    if args.display:
        cv2.namedWindow('{}'.format(args.video))
    if args.stream:
        stream = launch_stream_hls(args.host,args.port)
        
    while(True):
        try:
        # Capture frame-by-frame
            ret, frame = cap.read()
            if args.display:
                cv2.imshow('{}'.format(args.video),frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            if args.stream:
                stream.stdin.write(frame.tobytes())
                
        except KeyboardInterrupt:
            print('User finished using CTRL+C')
            stream.stdin.close()
            stream.stderr.close()
            stream.wait()
            break
    
    cv2.destroyAllWindows()
    stream.stdin.close()
    stream.stderr.close()
    stream.wait()
    cap.release()
            
          