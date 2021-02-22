import os
import cv2
import ffmpeg
import numpy as np
from pathlib import Path

''' 
    These functions are for videos that are recorded from an external source like a go pro.
    The videos must be recorded using a light sync method. The functions take the raw videos
    and concats them (if necessary) and trims the videos to the frame where the flash occurs.
'''


class SyncVids:
    def concatVideos(Source_video_folder, concatPath):
        '''Functions input is filepath is path to raw video folder
        If the videos in the folder are multiple parts the function uses ffmpeg to concat the video parts together
        It saves the concated video to an output folder 
        '''
        
        concatFolder = Source_video_folder+'/Concat' 
        videoList = os.listdir(Source_video_folder)
    
        multipleParts = True
        #Create a txt file for names of video parts
        txtFileList = []
        camNameList = []
        for ii in range(self.num_of_cameras): 
            #camNameTxt = open(Source_video_folder+'/cam'+str(ii+1)+'vids.txt','a')
            #txtFileList.append(camNameTxt)
            camName = 'Cam'+str(ii+1)
            camNameList.append(camName)
        numOfParts = len(videoList)/self.num_of_cameras
        k = 0
        for video in os.listdir(Source_video_folder):  #for loop parses through the video folder 
            if video[:4] in camNameList:
                x = open(Source_video_folder+'/'+video[:4]+'vids.txt','a')
                #idx = camNameList.index(video[:4])
                x.write('file'+" '" +'\\'+video+"'")
                x.write('\n')
            k+=1
        #Use ffmpeg to join all parts of the video together
        in_file= ffmpeg.input
        for jj in range(self.num_of_cameras):
            (ffmpeg
            .input(Source_video_folder+'/Cam'+str(jj+1)+'vids.txt', format='concat', safe=0)
            .output(concatPath+'/'+camNameList[jj][:4]+'.mp4', c='copy')
            .run()
            )

    def trimVideos(Inputfilepath,syncPath,numCams):
        '''Function input is the filepath for undistorted videos and a filepath for the desired output path
        The function finds the frame at the beginning and end of the video where a light flash occurs 
        The video is then trimmed based on those frame numbers
        Outputs the trimmed video to specified filepath
        '''    
        RawFilePath = Inputfilepath +'/RawVideos'
        if not os.path.exists(syncPath):
            os.mkdir(syncPath)
        videoList = os.listdir(RawFilePath)    
        for ii in range(numCams):
            
            vidcap = cv2.VideoCapture(RawFilePath+'/'+videoList[ii])#Open video
            vidWidth  = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH) #Get video height
            vidHeight = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT) #Get video width
            video_resolution = (int(vidWidth),int(vidHeight)) #Create variable for video resolution
            vidLength = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
            vidfps = vidcap.get(cv2.CAP_PROP_FPS)
            success,image = vidcap.read() #read a frame
            maxfirstGray = 0 #Intialize the variable for the threshold of the max brightness of beginning of video
            maxsecondGray = 0 #Intialize the variable for the threshold of the max brightness of end of video
            
            for jj in range(int(vidLength)):#For each frame in the video
                
                success,image = vidcap.read() #read a frame
                if success: #If frame is correctly read
                    if jj < int(vidLength/3): #If the frame is in the first third of video
                        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #Convert image to greyscale
                        if np.average(gray) > maxfirstGray:#If the average brightness is greater than the threshold
                            maxfirstGray = np.average(gray)#That average brightness becomes the threshold
                            firstFlashFrame = jj#Get the frame number of the brightest frame
                    if jj > int((2*vidLength)/3):
                        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #Convert image to greyscale
                        if np.average(gray) > maxsecondGray:#If the average brightness is greater than the threshold
                            maxsecondGray = np.average(gray)#That average brightness becomes the threshold
                            secondFlashFrame = jj #Get the frame number of the brightest frame
                else:#If the frame is not correctly read
                    continue#Continue
            input1 = ffmpeg.input(RawFilePath+'/'+videoList[ii])#input for ffmpeg

            node1_1 = input1.trim(start_frame=firstFlashFrame,end_frame=secondFlashFrame).setpts('PTS-STARTPTS')#Trim video based on the frame numbers
            #syncVid = Path(syncPath, videoList[ii])
            syncVid = syncPath + '/'+ videoList[ii]
            node1_1.output(syncVid).run()#Save to output folder