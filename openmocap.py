#built-in python libraries
import glob
import json
import os
import pickle
import re
import subprocess
import time
import datetime
from pathlib import Path
from functools import reduce 
#import webcam

#non-built-in python libraries
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.progress import track
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


from SyncVids import SyncVids



######################################################################################################################
####
####
####
####      ███████ ██    ██ ███    ██  ██████ ████████ ██  ██████  ███    ██  
####      ██      ██    ██ ████   ██ ██         ██    ██ ██    ██ ████   ██  
####      █████   ██    ██ ██ ██  ██ ██         ██    ██ ██    ██ ██ ██  ██  
####      ██      ██    ██ ██  ██ ██ ██         ██    ██ ██    ██ ██  ██ ██  
####      ██       ██████  ██   ████  ██████    ██    ██  ██████  ██   ████  
####
####
####
######################################################################################################################
##### The actual OpenMoCap methods!
##### I hate that this is on top of this file :(
##### This should be in a different file, because I personally believe that the actual
##### "runned" code shold be at the TOP of files,
##### but this is a simpler solution until I figure out how to Python
######################################################################################################################
######################################################################################################################

class Session: #session like "recording session"

    def __init__(self):
        self.sessionID = '' #The sessionID tag will be used to generate files names and whatnot
        self.baseFolder = '' #The folder where the to-be-processed videos live (in a folder called "synced Vids")
        self.DLCconfigPath = '' #Filepath of where the config file for DLC videos is 
        self.numCams = ''#The number of cameras used in this recording session
        self.openPoseDataPath    = ''#Where the open pose data lives
        self.dlcDataPath    = ''#Where the DLC data lives
    ######################################################################################################################
    ###
    ###    ██████  ██████  ███████ ███    ██     ██████   ██████  ███████ ███████     ███████ ████████ ██    ██ ███████ ███████ 
    ###   ██    ██ ██   ██ ██      ████   ██     ██   ██ ██    ██ ██      ██          ██         ██    ██    ██ ██      ██      
    ###   ██    ██ ██████  █████   ██ ██  ██     ██████  ██    ██ ███████ █████       ███████    ██    ██    ██ █████   █████   
    ###   ██    ██ ██      ██      ██  ██ ██     ██      ██    ██      ██ ██               ██    ██    ██    ██ ██      ██      
    ###    ██████  ██      ███████ ██   ████     ██       ██████  ███████ ███████     ███████    ██     ██████  ██      ██                                                                                                                           
    ###   
    ######################################################################################################################

    def RunOpenPose(self):
        '''The function takes the undistorted video and processes the videos in openpose
        The output is openpose overlayed videos and raw openpose data
        '''
        if not self.openPoseDataPath:
            self.openPoseDataPath  =  self.baseFolder / 'OpenPoseData' #the folder where openpose data lives

        if not self.openPoseDataPath.exists(): 
            os.mkdir(self.openPoseDataPath)

        # os.chdir(self.openPoseExePath)
        
        self.openPose_jsonPathList = [] #list to hold the paths to the json files
        self.openPose_imgPathList = []

        for thisVidPath in self.nSyncedVidFolder.iterdir():  #Run OpenPose ('Windows Portable Demo') on each video in the raw video folder
            if thisVidPath.suffix =='.mp4': #NOTE - build some list of 'synced video names' and check against that 
                print('OpenPosing: ', thisVidPath.name )
                vidPath = self.openPoseDataPath / thisVidPath.stem
                jsonPath = vidPath / 'json'
                imgPath =  vidPath / 'img'
                jsonPath.mkdir(parents=True, exist_ok=True) #this camera's json files (with keypoints)
                imgPath.mkdir(parents=True, exist_ok=True)   #this camera's image files (side note - pathlib is sick!)
                self.openPose_jsonPathList.append(jsonPath)
                self.openPose_imgPathList.append(imgPath)
                netRes = 320 #if you get an "out of memory" error, decrease this number (must be a multiple of 16)            
                # subprocess.call(['./bin/OpenPoseDemo.exe', '--video', str(thisVidPath),' --face ',' --hand ',  ' --write_json ', str(jsonPath),  ' --net_resolution -1x'+ str(netRes)], shell=True)
            else:
                print('Skipping: ', thisVidPath.name )
        f=9


    def ParseOpenPose(self):
        thisCamNum = -1

        ## %%
        #build header for dataframe - NOTE - #openpose data comes in a line ordered 'pixel x location (px)', 'pixel y (py)', 'confidence (conf)' for each keypoint  
        
        dataFrameHeader = []
        bodyCols = 75
        handCols = 63 #per hand
        faceCols = 210 #das alotta face!
        headerLength = bodyCols + 2*handCols + faceCols#should be 411 for whatever version of openpose i was using on 11 Jan 2021
        numImgPoints = headerLength

        for bb in range(0,int(bodyCols/3)): #loop through the number of body markers (i.e. #bodyCols/3)
            dataFrameHeader.append('body_' + str(bb).zfill(3) + '_pixx')
            dataFrameHeader.append('body_' + str(bb).zfill(3) + '_pixy')
            dataFrameHeader.append('body_' + str(bb).zfill(3) + '_conf')

        for hr in range(0,int(handCols/3)): #loop through the number of handR markers (i.e. #handCols/3)
            dataFrameHeader.append('handR_' + str(hr).zfill(3) + '_pixx')
            dataFrameHeader.append('handR_' + str(hr).zfill(3) + '_pixy')
            dataFrameHeader.append('handR_' + str(hr).zfill(3) + '_conf')

        for hl in range(0,int(handCols/3)): #loop through the number of handL markers (i.e. #handCols/3)
            dataFrameHeader.append('handL_' + str(hl).zfill(3) + '_pixx')
            dataFrameHeader.append('handL_' + str(hl).zfill(3) + '_pixy')
            dataFrameHeader.append('handL_' + str(hl).zfill(3) + '_conf')

        for ff in range(0,int(faceCols/3)): #loop through the number of Face markers (i.e. #faceCols/3)
            dataFrameHeader.append('face_' + str(ff).zfill(3) + '_pixx')
            dataFrameHeader.append('face_' + str(ff).zfill(3) + '_pixy')
            dataFrameHeader.append('face_' + str(ff).zfill(3) + '_conf')

        assert len(dataFrameHeader) == headerLength, ['Header is the wrong length! Should be ' +  str(headerLength) + ' but it is ' + str(len(dataFrameHeader)) + ' Check version of OpenPose?']

        ## %% 
        ## load in data from json files
        numFrames = int(len(list(self.openPose_jsonPathList[0].glob('*'))))
        numMarkers= int(int(len(dataFrameHeader)/3))
        numCams = int(self.numCams)

        openPoseData_nCams_nFrames_nImgPts_XYC = np.ndarray([numCams,numFrames,numMarkers,3]) #hardcoding for now because I am a bad person

        for thisCams_JsonFolderPath in track(self.openPose_jsonPathList, description='Parsing json\'s into a dataframe (per cam)' ):
            thisCamNum += 1
            # print('Parsing into a dataframe: ', thisCams_JsonFolderPath.name )
            jsonPaths = sorted(thisCams_JsonFolderPath.glob('*.json')) #glob is a "generator(?)" for paths to all the jason for THIS camara            
            
            frameNum = -1
            for thisJsonPath in jsonPaths: #loop throug all the json files and save their 'people' data to a dictionary (which will then be formatted into a pandas dataframe). NOTE - will be empty array if no hoomans visible in frame
                # print('loading: ', thisJsonPath.name)
                frameNum += 1 #frame number we're on
                thisJsonData = json.loads(thisJsonPath.read_bytes())
                thisJsonData = thisJsonData['people'] # #FEATURE_REQUEST -  at some point, we should check the openpose version (save it with the data somehow, verify everything is the same version, use different markernamlists for different versions, etc)

                if thisJsonData: #if this json has data
                    bodyData  = np.array(thisJsonData[0]['pose_keypoints_2d'])
                    handRData = np.array(thisJsonData[0]['hand_right_keypoints_2d'])
                    handLData = np.array(thisJsonData[0]['hand_left_keypoints_2d'])
                    faceData  = np.array(thisJsonData[0]['face_keypoints_2d'])
                    thisFrameRow = np.hstack((bodyData,handRData, handLData, faceData)) #horizontally concatenate these arrays                
                else: #if this json is empty, just stuff it fulla NaNs
                    thisFrameRow = np.empty([headerLength])
                    thisFrameRow.fill(np.nan)


                assert thisFrameRow.size == headerLength, ['Header is the wrong length! Should be ' +  str(headerLength) + ' but it is ' + str(thisFrameRow.size) + ' Check version of OpenPose?']
                
                openPoseData_nCams_nFrames_nImgPts_XYC[thisCamNum, frameNum, :, :] = np.reshape(thisFrameRow, [137,3]) #hard coding b/c I'm a bad person
                
            
            
        
        self.openPoseData_nCams_nFrames_nImgPts_XYC = openPoseData_nCams_nFrames_nImgPts_XYC
        self.dataFrameHeader = dataFrameHeader
        self.numImgPoints = numImgPoints
        self.numFrames = frameNum  #NOTE - Need to find a safer way to get this number

    # def ReconstructOpenPoseSkeleton(self):
    #     #hacky for now#

    #     nCamDataFrames = self.openPoseDataFrames_nCams
    #     openPoseNParrays_nCams = self.openPoseNParrays_nCams
        
    #     #triple nested for-loops baby! Now we're coding with gas! (but like, the fart kind of gas)
    #     #desired data shape - (numCams, numImagePoints, numFrames, numDim(2))
        
    #     numCams = self.numCams
    #     numImgPoints = self.numImgPoints
    #     numFrames = self.numFrames
    #     openPoseData_cam_imgpt_fr_dim = np.ndarray([numCams,numImgPoints, numFrames,2])

    #     for thisCamNum in range(numCams):
    #         thisCamDataFrame = nCamDataFrames[thisCamNum]            
    #         for thisImgPointNum in range(numImgPoints):
    #             thisImgPoint = thisCamArray[thisImgPointNum,:]
    #             for thisFrameNum in range(numFrames):
    #                 for thisDimNum in range(2):
    #                     f =9

    ######################################################################################################################################  
    ######################################################################################################################################  
    ####
    ####
    #### ██████  ███████ ███████ ██████  ██       █████  ██████   ██████ ██    ██ ████████     ███████ ████████ ██    ██ ███████ ███████ 
    #### ██   ██ ██      ██      ██   ██ ██      ██   ██ ██   ██ ██      ██    ██    ██        ██         ██    ██    ██ ██      ██      
    #### ██   ██ █████   █████   ██████  ██      ███████ ██████  ██      ██    ██    ██        ███████    ██    ██    ██ █████   █████   
    #### ██   ██ ██      ██      ██      ██      ██   ██ ██   ██ ██      ██    ██    ██             ██    ██    ██    ██ ██      ██      
    #### ██████  ███████ ███████ ██      ███████ ██   ██ ██████   ██████  ██████     ██        ███████    ██     ██████  ██      ██      
    ####
    ####
    ######################################################################################################################################  
    ######################################################################################################################################                                                                                                                                  
    '''                                                                                                                    
    def ParseDLCdata(self): #note this is all sloppy af just to get a completed video together. Latter iterations will use DLC methods and be smarter about Pandas Dataframs and whatnot
        

        if not self.dlcDataPath:
            self.dlcDataPath  =  self.baseFolder / 'DLCdata' / 'videos' #the folder where DLC data lives (except this won't work because the actual data is in [thatPath]/[nameofmodelthing/videos]), but don't tell anyone

        dlcCSVPaths = self.dlcDataPath.glob('*.csv') #NOTE - Super hacky here. Need to fix these methods (replace with DLC native functions?)
        
        numCams = int(self.numCams)
        numFrames = self.numFrames+1
        
        if self.sessionID == 'test6_01_21a': #NOTE - THIS IS DUMB AND BAD
            numCols =  8 #this is the number of tracked points in DLC - hardcoding for now because I am a bad person 
        else:
            numCols = 3 #NOTE - AGAIN< THIS IS DUMB AND BAD >

        dlc_nCams_nFrames_nImgPts_XYC = np.ndarray([numCams, numFrames, numCols, 3])
        dlc_nCams_nFrames_nImgPts_XYC.fill(np.nan)
        camNum = -1
        for thisCSVpath in dlcCSVPaths:     #triple nested for loop so I don't have to figure out np.reshape lol
            camNum += 1
            thisCam_dlcDataFrame = pd.read_csv(thisCSVpath, skiprows=1, header = [0,1]) #NOTE - This is dumb and bad. No need for this pandas dataframe, I think 
            thisCam_dlcNumpy = thisCam_dlcDataFrame.to_numpy()
            thisCam_dlcNumpy = thisCam_dlcNumpy[:, 1:] #remove first column (frame numbers)

            for thisFrame in range(thisCam_dlcNumpy.shape[0]):
                for thisImgPt in range(0,thisCam_dlcNumpy.shape[1],3):
                    dlc_nCams_nFrames_nImgPts_XYC[camNum,thisFrame,int(thisImgPt/3), 0] = thisCam_dlcNumpy[thisFrame, thisImgPt]
                    dlc_nCams_nFrames_nImgPts_XYC[camNum,thisFrame,int(thisImgPt/3), 1] = thisCam_dlcNumpy[thisFrame, thisImgPt+1]
                    dlc_nCams_nFrames_nImgPts_XYC[camNum,thisFrame,int(thisImgPt/3), 2] = thisCam_dlcNumpy[thisFrame, thisImgPt+2]
        

        if self.debug:
            fig = plt.figure()
            
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224)

            cam1im = self.firstImage_nCams_list[0]
            cam2im = self.firstImage_nCams_list[1]
            cam3im = self.firstImage_nCams_list[2]
            cam4im = self.firstImage_nCams_list[3]
            
            ax1.imshow(cam1im)
            ax1.plot(dlc_nCams_nFrames_nImgPts_XYC[0,0,:,0],dlc_nCams_nFrames_nImgPts_XYC[0,0,:,1])

            ax2.imshow(cam2im)
            ax2.plot(dlc_nCams_nFrames_nImgPts_XYC[1,0,:,0],dlc_nCams_nFrames_nImgPts_XYC[1,0,:,1])

            ax3.imshow(cam3im)
            ax3.plot(dlc_nCams_nFrames_nImgPts_XYC[2,0,:,0],dlc_nCams_nFrames_nImgPts_XYC[2,0,:,1])

            ax4.imshow(cam4im)
            ax4.plot(dlc_nCams_nFrames_nImgPts_XYC[3,0,:,0],dlc_nCams_nFrames_nImgPts_XYC[3,0,:,1])

            plt.show()
        
        self.dlc_nCams_nFrames_nImgPts_XYC = dlc_nCams_nFrames_nImgPts_XYC'''
    def runAndParseDeepLabCut(Inputfilepath,OutputFilepath, ConfigPath):
        '''Function inputs are filepath to videos to be tracked by DLC and the folder to save the output to
        Videos are copied to output folder, than processed in DLC based on the dlc config path 
        DLC output is saved in outputfilepath and the output is also converted to npy and saved as well
        '''
        
        #####################Copy Videos to DLC Folder####################          This step may not be necessary?
        for dir in [Inputfilepath]:#Iterates through input folder
            for video in os.listdir(dir):#Iterates through each video in folder
                #ffmpeg call to copy videos to dlc folder
                subprocess.call(['ffmpeg', '-i', Inputfilepath+'/'+video,  OutputFilepath+'/'+video])


        #################### DeepLabCut ############################
        for dir in [OutputFilepath]:# Loop through dlc folder
            for video in os.listdir(dir):
                #Analyze the videos through deeplabcut
                deeplabcut.analyze_videos(ConfigPath, [OutputFilepath +'/'+ video], save_as_csv=True)
                
        #####################################################MAKE SURE THIS WORKS IT WAS GIVNG TROUBLE LAST TIME YOU USED THIS SCRIPT
        for dir in [OutputFilepath]:#Loop through dlc folder 
            for video in dir:# for each video in folder
                #Create a DLC video   
                deeplabcut.create_labeled_video(ConfigPath, glob.glob(os.path.join(OutputFilepath ,'*mp4')))

        #If there is not a folder for dlc npy output, create one
        if not os.path.exists(OutputFilepath + 'DLCnpy'):
            os.mkdir(OutputFilepath+ 'DLCnpy')
        
        #Load all dlc csv output files  
        csvfiles = glob.glob(OutputFilepath+'/*csv')
        #For loop gets csv data from all cameras
        j=0
        for data in csvfiles:     
            datapoints = pd.read_csv(data) # read in the csv data 
            print(datapoints)            

            parsedDlcData = datapoints.iloc[3:,7:10].values#the last element in the array is the P value
            #print(parsedDlcData)
        
            print(parsedDlcData)
            np.save(OutputFilepath+'/DLCnpy/dlc_'+cam_names[j]+'.npy',parsedDlcData)#Save data
            j+=1

    ####################################################################################################################################  
    ###  
    ###   ██████  █████  ██      ██ ██████       ██████  █████  ██████  ████████ ██    ██ ██████  ███████     ██    ██  ██████  ██      ██    ██ ███    ███ ███████ 
    ###  ██      ██   ██ ██      ██ ██   ██     ██      ██   ██ ██   ██    ██    ██    ██ ██   ██ ██          ██    ██ ██    ██ ██      ██    ██ ████  ████ ██      
    ###  ██      ███████ ██      ██ ██████      ██      ███████ ██████     ██    ██    ██ ██████  █████       ██    ██ ██    ██ ██      ██    ██ ██ ████ ██ █████   
    ###  ██      ██   ██ ██      ██ ██   ██     ██      ██   ██ ██         ██    ██    ██ ██   ██ ██           ██  ██  ██    ██ ██      ██    ██ ██  ██  ██ ██      
    ###   ██████ ██   ██ ███████ ██ ██████  ██   ██████ ██   ██ ██         ██     ██████  ██   ██ ███████       ████    ██████  ███████  ██████  ██      ██ ███████ 
    ###                                                                                                                                                                                              
    ###                                                                                                                                                                                              
    ####################################################################################################################################            

    def CalibrateCaptureVolume(self):

        ###########
        ##STEP 1  - pull sample frame from each video (for calibration)        
        ###########

        firstImage_nCams_list = [] #list to hold the first image from each camera
        
        for thisVidPath in self.nSyncedVidFolder.glob('*.mp4'): 
            vidcap = cv2.VideoCapture(str(thisVidPath))
            success,image = vidcap.read()           
            assert success #this is the most inspirational line of code I've ever written :')
            #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) 
            firstImage_nCams_list.append(image)

        self.firstImage_nCams_list = firstImage_nCams_list #save for posterity and latter debugging
        numCams = len(firstImage_nCams_list)
        self.numCams = numCams
        ###########
        ##STEP 2  - Detect "charuco" chessboard and corners
        ###########

        allCorners = [] #corners from all cameras
        allIds = []     #id's of the corners in 'allCorners'
        allIdSets = []
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001) #sub pixel corner detection criterion
        
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250) #dictionary containing 250 4x4 Aruco markers
        board = cv2.aruco.CharucoBoard_create(7, 5, 1, .8, aruco_dict)    #Charuco chessboard (must match what was used inthe recording!!)
        imboard = board.draw((2000, 2000)) #image of calib board
        camNum = 0

        for thisIm in firstImage_nCams_list:
            camNum += 1
            thisGrey = cv2.cvtColor(thisIm,cv2.COLOR_RGB2GRAY) 
            corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(thisGrey, aruco_dict) #detect corners and get ID's of each detected corners (based on aruco_dict)
            
            
            if corners: #if corners is not empty, do the subpixel resolution thing
                for corner in corners:  #refines detected points for subpixel accuracy (supposedly. I am dubious that this is useful)
                    cv2.cornerSubPix(thisGrey, corner, winSize = (3,3), zeroZone = (-1,-1), criteria = criteria)

                res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,thisGrey,board)  #seriously. Why tf would you call it "res2" in the f-ing TUTORIAL. Just name the f-ing variable something MEANINGFUL for f'sake

                if res2[0]>0: #if res2[0] isn't zero, then the system found corners of the charuco board
                    # allCorners.append(np.squeeze(res2[1]))      
                    # allIds.append(np.squeeze(res2[2]))
                    theseCharucoPts = (res2[1]) #if you np.squeeze this, the Charuco fuction below fails (WHICH IS SO DUMB!)
                    theseCharucoIds = (res2[2])
                    allCorners.append(theseCharucoPts)
                    allIds.append(theseCharucoIds)                    
                else:
                    print(str(camNum), ' didn\'t return any charcuo corners!')
        

        allCornersOG = np.array(allCorners, dtype=object) #keep these around for debuggibg
        sharedIds = reduce(np.intersect1d, allIds) #only keep corner ids that are seen by all cameras

        for camNum in range(numCams):    
            mask = ~np.in1d(allIds[camNum], sharedIds) #find elements in this cam's corners that are not seen by all cameras
            allCorners[camNum] = np.delete(allCorners[camNum], mask, axis=0) #delete corners that aren't seen by all camers
            allIds[camNum] = np.array(sharedIds) #keep track of ids

        if self.debug:
            camNum = -1
            for thisIm in firstImage_nCams_list:
                #plt.ion()
                camNum += 1
                pts = np.squeeze(allCorners[camNum])
                ptsOG = np.squeeze(allCornersOG[camNum])
                
                xOG = ptsOG[:,0] #original detected corners
                yOG = ptsOG[:,1]
                
                x = pts[:,0]#'filtered' corners (only corners seen by all cameras should remain)
                y = pts[:,1]

                fig1 = plt.figure(num=392, figsize=(8,6))
                plt.subplot(2,2,camNum+1)
                plt.imshow(cv2.cvtColor(thisIm, cv2.COLOR_BGR2RGB))
                plt.plot(xOG,yOG,'m.')
                plt.plot(x,y, 'go',markerfacecolor='none')
                
                if camNum == len(firstImage_nCams_list)-1:
                    plt.pause(0.01)
                    plt.show()


            
        f=9

        imsize = thisIm.shape[0:2] #sloppy, do this earlier


        ###########
        ##STEP 3  - Use detected charuco board points to determine camera positions (cam extrinsics) and intrinsics (though I don't trust those numbers)
        ###########
        print("Calibrating camera positions!")

        cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                    [    0., 1000., imsize[1]/2.],
                                    [    0.,    0.,           1.]])

        distCoeffsInit = np.zeros((5,1))
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
        
        (ret, camera_matrix, distortion_coefficients0,
        rotation_vectors, translation_vectors,
        stdDeviationsIntrinsics, stdDeviationsExtrinsics,
        perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                        charucoCorners=allCorners,
                        charucoIds=allIds,
                        board=board,
                        imageSize=imsize,
                        cameraMatrix=cameraMatrixInit,
                        distCoeffs=distCoeffsInit,
                        flags=flags,
                        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
        #ret - No clue what this variable is, and I can't find mention of it in the docs :-/
        #camera_matrix - Multiply your image by this to move the origin from the top left corner (image coordinates) to the center of the image (not sure what the 1000's are doing on the diagonal though)
        #distortion_coefficients0 = lense distortion coeficients (i.e. camera intrinsics) - I don't really trust these numbers
        #rotation vectors - euler angle rotations for each camera (relative to the principle camera, which is Cam1?)
        #translation_vectors = XYZ position of each camera (i.e. nodal point of each lense?). I *think* units are in the size of the charuco board squares
        # and then a couple variables reporting errors of various kinds, I guess
        translation_vectors = np.squeeze(np.array(translation_vectors))
        rotation_vectors = np.squeeze(np.array(rotation_vectors))
        camera_matrix = camera_matrix #assume all cameras have same resulotion (and thereby same camera_matrix) for now.

        camRotMat = np.ndarray([numCams, 3, 3]) #gonna stuff this bad boi fulla 3x3 rotation matricies derived from rotation_vectors
        camHomoMats = np.ndarray([numCams,4,4]) #this'll be the 'homogeneous matrix' (need to brush up on what this actually is)
        camInvHomoMats = camHomoMats #this'll hold the inverted homogenous matrices
        camTransMats = np.ndarray([numCams,3,4])
        camProjMats  = np.ndarray([numCams,3,4]) #<-- This is the magical matrix that projects points from the camera image out into the world! I think?!

        for cc in range(numCams): 
            #get homogenous matrix (and inverse Homo) for each camera! 
            camRotMat[cc],_ = cv2.Rodrigues(rotation_vectors[cc])
            thisHomoMat = np.zeros((4,4))
            thisHomoMat[:3,:3],thisHomoMat[:3,-1] = camRotMat[cc],translation_vectors[cc].reshape(3,)
            thisHomoMat[-1,:] = np.array([0,0,0,1])
            camHomoMats[cc] = thisHomoMat
            
            #invert Homogenous matrix (I don't understand this math, I'm just copying what Yifan did)
            trans_R = thisHomoMat[:3,:3].T#why transpose (why indeed?)
            trans_T = -trans_R.dot(thisHomoMat[:3,-1])
            inv_H = np.zeros((4,4))
            inv_H[:3,:3] = trans_R
            inv_H[:3,-1] = trans_T
            inv_H[-1,-1] = 1
            camInvHomoMats[cc] = inv_H

            #get transformation matrix of each camera relatie to the principle camra (again, just copying Yifan's work)
            if cc == 0: #make the first camera te principle camera
                invH0 = inv_H  #the first homogenies transformation matrixs is used to calculate the relative position to the principle camera(first camera)
                camTransMats[cc] = np.hstack((np.eye(3, 3), np.zeros((3, 1))))  #the principle camera have projection matrix with rotation matrix to be a diagonal matrix, translation vector to be (0,0,0)
            else:
                thisHomoMat_rel = thisHomoMat.dot(invH0) #relative position of this cmaera to the principle camera
                thisRot_rel = thisHomoMat_rel[:3,:3] #pull out RotMat
                thisTrans_rel = thisHomoMat_rel[:3,-1].reshape(3,1) #pull out XYZ Trans Vec
                camTransMats[cc] = np.hstack((thisRot_rel, thisTrans_rel))    #stuff 'em together to tget the translation matrix for this camera (relative to principle cam) :D
            
            #Calc projection matrices for each camera
            camProjMats[cc] = np.dot(camera_matrix, camTransMats[cc])
        
        
        self.camProjMats = camProjMats
        self.camTransMats = camTransMats
        self.camera_matrix = camera_matrix
        self.imsize = imsize
        charuco_nCams_nFrames_nImgPts_XYC = np.ndarray([numCams, 1, len(allCorners[0]), 3])
        for camNum in range(numCams):
            charuco_nCams_nFrames_nImgPts_XYC[camNum,0,:,0:2] = np.squeeze(allCorners[camNum])

        charuco_nCams_nFrames_nImgPts_XYC[:,:,:,2] = 1

        charucoCornersXYZ = self.Reconstruct3D(charuco_nCams_nFrames_nImgPts_XYC) #tranigulate points!!

        if sesh.debug:
            fig = plt.figure()
            ax = fig.add_subplot(111,projection='3d')
            x = charucoCornersXYZ[:,0]
            y = charucoCornersXYZ[:,1]
            z = charucoCornersXYZ[:,2]
            mx = np.mean(x)
            my = np.mean(y)
            mz = np.mean(z)
            ax.set_xlim([mx-3, mx+3])
            ax.set_ylim([my-3, my+3])
            ax.set_zlim([mz-3, mz+3])
            ax.scatter(x,y,z, marker='o')
            plt.pause(0.01)
            plt.show()

        self.charucoCornersXYZ = charucoCornersXYZ
            
    ##############################################################################################################################################################################
    ####    
    ####  
    ####  ██████  ███████  ██████  ██████  ███    ██ ███████ ████████ ██████  ██    ██  ██████ ████████     ██████  ██████      ██████  ████████ ███████ 
    ####  ██   ██ ██      ██      ██    ██ ████   ██ ██         ██    ██   ██ ██    ██ ██         ██             ██ ██   ██     ██   ██    ██    ██      
    ####  ██████  █████   ██      ██    ██ ██ ██  ██ ███████    ██    ██████  ██    ██ ██         ██         █████  ██   ██     ██████     ██    ███████ 
    ####  ██   ██ ██      ██      ██    ██ ██  ██ ██      ██    ██    ██   ██ ██    ██ ██         ██             ██ ██   ██     ██         ██         ██ 
    ####  ██   ██ ███████  ██████  ██████  ██   ████ ███████    ██    ██   ██  ██████   ██████    ██        ██████  ██████      ██         ██    ███████ 
    ####                                                                                                                                                                                                                                                                                             
    ####                                                                                                                                                                         
    ##############################################################################################################################################################################
                                                                                                                                                                     

    def Reconstruct3D(self, imgPoints_nCams_nFrames_nImgPts_XYC): #renamed 'triangulate' from Yifan's code - (was a class, it's now a method of 'Session')  #Note - I don't really understands the math (yet!), I'm just copying from Yifan
        """
        ImgPoints: a numpy array of dimensions - (numCams, numFrames, numImgPoints, numDims(3, X,Y,Confidence)) 
        self.camProjMats: a numCamsx3x4 matrix, numCams is number of views, each view has an associated 3x4 projection matrix
        """
        camProjMats = self.camProjMats
        numCams = imgPoints_nCams_nFrames_nImgPts_XYC.shape[0]
        numFrames = imgPoints_nCams_nFrames_nImgPts_XYC.shape[1]
        numImgPoints = imgPoints_nCams_nFrames_nImgPts_XYC.shape[2]
        

        if numCams != camProjMats.shape[0]:
            raise Exceptions('number of views must be equal to number of projection matrix')
        
        svdMatA = np.ndarray([numCams*2,4]) #prepare svd matrix A        
        reconstructed3Dpoints = np.ndarray([numFrames, numImgPoints,3])

        for thisFrameNum in track(range(numFrames), description='Reconstructing 3D points from Pixel Data!'): #for each frame

            for thisImgPointNum in range(numImgPoints):
                #skipping 'confidence' check for now - Later, don't do this for points that come in with low confidence values
                for thisCamNum in range(numCams): #for each view
                    
                    u_imgX = imgPoints_nCams_nFrames_nImgPts_XYC[thisCamNum, thisFrameNum, thisImgPointNum,  0]
                    v_imgY = imgPoints_nCams_nFrames_nImgPts_XYC[thisCamNum, thisFrameNum, thisImgPointNum,  1] #initialize x,y points
                    
                    svdMatA = np.ndarray([numCams*2,4]) #prepare svd matrix A   

                    for col in range(4):
                        svdMatA[thisCamNum*2+0,col] = u_imgX*camProjMats[thisCamNum,2,col] - camProjMats[thisCamNum,0,col]
                        svdMatA[thisCamNum*2+1,col] = v_imgY*camProjMats[thisCamNum,2,col] - camProjMats[thisCamNum,1,col]

                # #Remove NaNs (i.e. camera views without data)
                # keepTheseRows = ~np.any(np.isnan(svdMatA), axis=1) #find rows that dont have nans in them
                # svdMatA = svdMatA[keepTheseRows,:]

                #Check to make sure you still have at least 2 camera views (i.e. at least the SVD thingo has at least 4 rows)
                # if svdMatA.shape[0] < 4:
                if np.sum(np.isnan(svdMatA)) >0: #skip all nans for now
                    P = [np.nan, np.nan, np.nan] #less than 2 views - Fill it with NaNs!
                else:                    
                    #SVD Magic! (I don't understand this bit)
                    U,s,V = np.linalg.svd(svdMatA)
                    P = V[-1,:] / V[-1,-1]           

                reconstructed3Dpoints[thisFrameNum,thisImgPointNum] = P[:3]
        
        return np.squeeze(reconstructed3Dpoints)

    ############################################################################################################################################################
    ############################################################################################################################################################
    ######     
    ######     ██████  ██       █████  ██    ██     ███████ ██   ██ ███████ ██      ███████ ████████  ██████  ███    ██     ██ 
    ######     ██   ██ ██      ██   ██  ██  ██      ██      ██  ██  ██      ██      ██         ██    ██    ██ ████   ██     ██ 
    ######     ██████  ██      ███████   ████       ███████ █████   █████   ██      █████      ██    ██    ██ ██ ██  ██     ██ 
    ######     ██      ██      ██   ██    ██             ██ ██  ██  ██      ██      ██         ██    ██    ██ ██  ██ ██        
    ######     ██      ███████ ██   ██    ██        ███████ ██   ██ ███████ ███████ ███████    ██     ██████  ██   ████     ██                                                                                                                                                                                                                            
    ######                                                                                                                                                                                                                      
    ############################################################################################################################################################
    ##########################################################################################################################################################                                                                                             
                                                                                             




    def PlaySkeleton(self, vidType):

        # vidType = 0 #No Cam Ims
        # vidType = 1 #Only Cam1 im
        # vidType = 2 #All Cam Ims

        #where to put the output images (later will be justa video)
        imOutPath = self.baseFolder / 'imOut'
        imOutPath.mkdir(parents=True, exist_ok=True)
        
        os.chdir(imOutPath)

        self.imOutPath = imOutPath

        skel_fr_mar_dim = self.skel_fr_mar_dim
        dlcPts_fr_mar_dim = self.dlcPts_fr_mar_dim

        dlc_nCams_nFrames_nImgPts_XYC = self.dlc_nCams_nFrames_nImgPts_XYC

        dlcCam1_fr_mar_xyc = np.squeeze(dlc_nCams_nFrames_nImgPts_XYC[0])
        dlcCam2_fr_mar_xyc = np.squeeze(dlc_nCams_nFrames_nImgPts_XYC[1])
        dlcCam3_fr_mar_xyc = np.squeeze(dlc_nCams_nFrames_nImgPts_XYC[2])
        dlcCam4_fr_mar_xyc = np.squeeze(dlc_nCams_nFrames_nImgPts_XYC[3])
              
        #pull out charuco board points
        charuco_x = self.charucoCornersXYZ[:,0]
        charuco_y = self.charucoCornersXYZ[:,1]
        charuco_z = self.charucoCornersXYZ[:,2]
        
        # define Skeleton connections 
        # head = [17, 15, 0, 16, 18]
        head = [ 15, 0, 16]
        spine = [0,1,8]
        rArm = [17, 15, 0, 16, 18]
        rArm = [4 ,3 ,2 ,1]
        lArm = [1, 5, 6, 7]
        rLeg = [11 ,10, 9, 8]
        lLeg = [14 ,13 ,12, 8]
        rFoot = [11, 23,22, 11, 24]
        lFoot = [14, 20, 19, 14, 21]

        # #define face parts
        # jaw = [0:16]
        # rEyeSmall = [36:41, 36]
        # lEyeSmall = [42:47, 42]
        # rBrow = [17:21]
        # lBrow = [22:26]
        # noseRidge = [27:30]
        # noseBot = [31:35]
        # mouthOut = [48:59, 48]
        # mouthIn = [60:67, 60]
        # rPup = 68
        # lPup = 69

        #Make some handy maps ;D
        rHandIDstart = 25 
        lHandIDstart = rHandIDstart+21

        thumb = np.array([0,1,2,3,4])
        index = np.array([0, 5,6,7,8])
        bird = np.array([0, 9,10,11,12])
        ring = np.array([0, 13,14,15,16])
        pinky = np.array([0, 17,18,19,20])

        #maps for dlc data
        wobbleBoard = [0,1,2,3,4,0,3,1,3,2,1,4,3,4] #that oughta cover it :P
        orangeBall  = dlcPts_fr_mar_dim[:,5,:]
        pinkBall    = dlcPts_fr_mar_dim[:,6,:]
        greenBall   = dlcPts_fr_mar_dim[:,7,:]
        
        ballTrailLen = 4

        
        #set axis limits based on charuco board
        mx = np.mean(charuco_x)
        my = np.mean(charuco_y)
        mz = np.mean(charuco_z)

        #paths to camera images (openposed)
        cam1imgPathList = list(sorted(self.openPose_imgPathList[0].glob('*.png')))
        cam2imgPathList = list(sorted(self.openPose_imgPathList[1].glob('*.png')))
        cam3imgPathList = list(sorted(self.openPose_imgPathList[2].glob('*.png')))
        cam4imgPathList = list(sorted(self.openPose_imgPathList[3].glob('*.png')))

        #set up figure
        fig = plt.figure(figsize=([11,8]))

        if vidType==0:
            axMain = fig.add_subplot(111,projection='3d')
        elif vidType == 1:
            axMain = fig.add_subplot(121,projection='3d')
            axCam1 = fig.add_subplot(122)
        elif vidType == 2:
            figGridSpec = fig.add_gridspec(2,4)
            axMain = fig.add_subplot(figGridSpec[:2,:2],projection='3d')
            axCam1 = fig.add_subplot(figGridSpec[0,2])
            axCam2 = fig.add_subplot(figGridSpec[0,3])
            axCam3 = fig.add_subplot(figGridSpec[1,2])
            axCam4 = fig.add_subplot(figGridSpec[1,3])
            
        axMain.view_init(azim = -90, elev=-75)
        plt.ion() #this makes it so the figure doesn't block the code from running (but it's weird and I don't fully understand it)
        startFrame = 120  
        endFrame = self.numFrames


        ##############################################################################################
        ##Frame-by-frame animation loop starts here
        ##############################################################################################

        for fr in range(startFrame, endFrame):
            fig.suptitle([self.sessionID, ' - Frame# - ', str(fr)])

            axMain.cla()

            if vidType ==1:
                axCam1.cla()
            elif vidType == 2:
                axCam1.cla()
                axCam2.cla()
                axCam3.cla()
                axCam4.cla()

            # pull out skel and dlc data for this frame

            sk_x = skel_fr_mar_dim[fr,:,0] #skeleton x data
            sk_y = skel_fr_mar_dim[fr,:,1] #skeleton y data
            sk_z = skel_fr_mar_dim[fr,:,2] #skeleton z data

            dlc_x = dlcPts_fr_mar_dim[fr,:,0]
            dlc_y = dlcPts_fr_mar_dim[fr,:,1]
            dlc_z = dlcPts_fr_mar_dim[fr,:,2]

            #plot charuco board points
            axMain.scatter(charuco_x,charuco_y,charuco_z, marker='o')

            #plot skeleton points
            axMain.scatter(sk_x,sk_y,sk_z, marker='.',color = 'k', s=4.)

            #plot skeleton connecting lines
            axMain.plot(sk_x[head],sk_y[head],sk_z[head], linestyle='-', color='g', linewidth = 1.)
            axMain.plot(sk_x[spine],sk_y[spine],sk_z[spine], linestyle='-', color = 'g', linewidth = 1.)
            axMain.plot(sk_x[rArm],sk_y[rArm],sk_z[rArm], linestyle='-', color = 'r', linewidth = 1.)
            axMain.plot(sk_x[lArm],sk_y[lArm],sk_z[lArm], linestyle='-', color = 'b', linewidth = 1.)
            axMain.plot(sk_x[rLeg],sk_y[rLeg],sk_z[rLeg], linestyle='-', color = 'r', linewidth = 1.)
            axMain.plot(sk_x[lLeg],sk_y[lLeg],sk_z[lLeg], linestyle='-', color = 'b', linewidth = 1.)
            axMain.plot(sk_x[rFoot],sk_y[rFoot],sk_z[rFoot], linestyle='-', color = 'r', linewidth = 1.)
            axMain.plot(sk_x[lFoot],sk_y[lFoot],sk_z[lFoot], linestyle='-', color = 'b', linewidth = 1.)

            # plot handybois
            # right hand
            axMain.plot(sk_x[thumb+rHandIDstart],sk_y[thumb+rHandIDstart],sk_z[thumb+rHandIDstart], linestyle='-', color = 'r', linewidth = 1.)
            axMain.plot(sk_x[index+rHandIDstart],sk_y[index+rHandIDstart],sk_z[index+rHandIDstart], linestyle='-', color = 'r', linewidth = 1.)
            axMain.plot(sk_x[bird+rHandIDstart],sk_y[bird+rHandIDstart],sk_z[bird+rHandIDstart], linestyle='-', color = 'r', linewidth = 1.)
            axMain.plot(sk_x[ring+rHandIDstart],sk_y[ring+rHandIDstart],sk_z[ring+rHandIDstart], linestyle='-', color = 'r', linewidth = 1.)
            axMain.plot(sk_x[pinky+rHandIDstart],sk_y[pinky+rHandIDstart],sk_z[pinky+rHandIDstart], linestyle='-', color = 'r', linewidth = 1.)

            #left hand
            axMain.plot(sk_x[thumb+lHandIDstart],sk_y[thumb+lHandIDstart],sk_z[thumb+lHandIDstart], linestyle='-', color = 'b', linewidth = 1.)
            axMain.plot(sk_x[index+lHandIDstart],sk_y[index+lHandIDstart],sk_z[index+lHandIDstart], linestyle='-', color = 'b', linewidth = 1.)
            axMain.plot(sk_x[bird+lHandIDstart],sk_y[bird+lHandIDstart],sk_z[bird+lHandIDstart], linestyle='-', color = 'b', linewidth = 1.)
            axMain.plot(sk_x[ring+lHandIDstart],sk_y[ring+lHandIDstart],sk_z[ring+lHandIDstart], linestyle='-', color = 'b', linewidth = 1.)
            axMain.plot(sk_x[pinky+lHandIDstart],sk_y[pinky+lHandIDstart],sk_z[pinky+lHandIDstart], linestyle='-', color = 'b', linewidth = 1.)

            #plot dlc data                        
            axMain.plot(dlc_x[wobbleBoard], dlc_y[wobbleBoard], dlc_z[wobbleBoard], linestyle='-', color = 'k', linewidth = .5)

            axMain.plot(orangeBall[fr,0], orangeBall[fr,1], orangeBall[fr,2], linestyle='none', marker='o',color = 'orange')
            axMain.plot(orangeBall[range(fr-ballTrailLen,fr+1),0], orangeBall[range(fr-ballTrailLen,fr+1),1], orangeBall[range(fr-ballTrailLen,fr+1),2], linestyle='-',color = 'orange')
            
            axMain.plot(pinkBall[fr,0], pinkBall[fr,1], pinkBall[fr,2], linestyle='none', marker='o',color = 'm')
            axMain.plot(pinkBall[range(fr-ballTrailLen,fr+1),0], pinkBall[range(fr-ballTrailLen,fr+1),1], pinkBall[range(fr-ballTrailLen,fr+1),2], linestyle='-',color = 'm')
            
            axMain.plot(greenBall[fr,0], greenBall[fr,1], greenBall[fr,2], linestyle='none', marker='o',color = 'green')
            axMain.plot(greenBall[range(fr-ballTrailLen,fr+1),0], greenBall[range(fr-ballTrailLen,fr+1),1], greenBall[range(fr-ballTrailLen,fr+1),2], linestyle='-',color = 'g')
            
             

            axMain.set_xlabel('x')
            axMain.set_ylabel('y')
            axMain.set_zlabel('z')

            axMain.set_xlim(xmin=mx-13, xmax=mx+13)
            axMain.set_ylim(ymin=my-26, ymax=my)
            axMain.set_zlim(zmin=mz-13, zmax=mz+13)


            # #show camera images?
            if vidType ==0:
                pass
            elif vidType == 1:
                axCam1.imshow(cv2.cvtColor(cv2.imread(str(cam1imgPathList[fr])),cv2.COLOR_BGR2RGB))
                axCam1.axis('off')
            elif vidType ==2:
                axCam1.imshow(cv2.cvtColor(cv2.imread(str(cam1imgPathList[fr])),cv2.COLOR_BGR2RGB))
                axCam1.plot(dlcCam1_fr_mar_xyc[fr,:,0], dlcCam1_fr_mar_xyc[fr,:,1], linestyle='none', marker = '.',color='w', markerfacecolor='none' )
                
                axCam2.imshow(cv2.cvtColor(cv2.imread(str(cam2imgPathList[fr])),cv2.COLOR_BGR2RGB))
                axCam2.plot(dlcCam2_fr_mar_xyc[fr,:,0], dlcCam2_fr_mar_xyc[fr,:,1], linestyle='none', marker = '.',color='w', markerfacecolor='none' )

                axCam3.imshow(cv2.cvtColor(cv2.imread(str(cam3imgPathList[fr])),cv2.COLOR_BGR2RGB))
                axCam3.plot(dlcCam3_fr_mar_xyc[fr,:,0], dlcCam3_fr_mar_xyc[fr,:,1], linestyle='none', marker = '.',color='w', markerfacecolor='none' )

                axCam4.imshow(cv2.cvtColor(cv2.imread(str(cam4imgPathList[fr])),cv2.COLOR_BGR2RGB))
                axCam4.plot(dlcCam4_fr_mar_xyc[fr,:,0], dlcCam4_fr_mar_xyc[fr,:,1], linestyle='none', marker = '.',color='w' , markerfacecolor='none')

                axCam1.axis('off')
                axCam2.axis('off')
                axCam3.axis('off')
                axCam4.axis('off')

            imName = [self.sessionID + '_frame' + str(fr-startFrame).zfill(6)]
            fig.savefig(str(fr-startFrame).zfill(6) , bbox_inches = 'tight') #save this frame out as a PNG until we (aka Aaron) figure out how to do python video writing
            plt.pause(0.01)
            plt.show()


            f = 9


class Exceptions(Exception):
    pass
        

            
############################################################################################
############################################################################################
####
#### ███    ███  █████  ██ ███    ██     ███████  ██████ ██████  ██ ██████  ████████ 
#### ████  ████ ██   ██ ██ ████   ██     ██      ██      ██   ██ ██ ██   ██    ██    
#### ██ ████ ██ ███████ ██ ██ ██  ██     ███████ ██      ██████  ██ ██████     ██    
#### ██  ██  ██ ██   ██ ██ ██  ██ ██          ██ ██      ██   ██ ██ ██         ██    
#### ██      ██ ██   ██ ██ ██   ████     ███████  ██████ ██   ██ ██ ██         ██                                                                                   
####                                                                                        
#############################################################################################
#############################################################################################                                                                                       


#######################################################
##### Initialize session class and set up paths
#######################################################

# plt.ion() #turn on interactive mode of pyplot (so figures don't block code. Use `plt.pause(0.01)` to force system to update figure)

sesh = Session()

sesh.debug = False



task = '' #set task to 'record' to use the webcams
cam_inputs = [1,2] #enter inputs as [input1, input2] i.e. [1,2,3,4]
sesh.numCams = 8

sesh.sessionID = "test6_01_21a"
if not sesh.sessionID: #if no custom ID is entered, one will be generated (as sesh[time of recording in hours:minutes:seconds]_[month]_[date])
    sesh.sessionID = datetime.datetime.now().strftime("sesh_%y_%m_%d_%H%M%S")
# sesh.DLCdataPath = Path(r"C:\Users\jonma\Dropbox\GitKrakenRepos\OpenMoCap\Data\test6_01_21a\DLCdata\z3ballWobbleBoard-JSM-2021-01-16\videos") #later iterations will use the DLC config.yaml to do this more intelligently

# sesh.sessionID = "sesh114858_01_29"
# sesh.DLCdataPath = Path(r"C:\Users\jonma\Dropbox\GitKrakenRepos\OpenMoCap\Data\sesh114858_01_29\DLCdata\OrangePinkGreenBall-JSM-2021-02-03\videos") #later iterations will use the DLC config.yaml to do this more intelligently

#sesh.seshFolder = Path.cwd()/'Data'/sesh.sessionID
#sesh.seshFolder.mkdir(exist_ok='True')#make a folder for the sessionID if none exists
sesh.baseFolder='C:/Users/chris/DLC/Axolotl/IncorrectNaming/20210217/Chonk'


#sesh.nSyncedVidFolder = Path(sesh.baseFolder, 'SyncedVideos')
sesh.nSyncedVidFolder = sesh.baseFolder+'/SyncedVideos'

sesh.openPoseExePath = Path("C:/openpose/")

########################################################
####### Sync Videos if Necessary      ################
######################################################

#SyncVids.concatVideos(sesh.baseFolder)     Currently not necessary but will be for long recordings
SyncVids.trimVideos(sesh.baseFolder, sesh.nSyncedVidFolder,sesh.numCams)


########################################################
##### Recording Stuff
########################################################


#------------------ROTATION DEFINITIONS
rotate0 = None
rotate90 = 90
rotate180 = 180
rotate270 = 270


# =============================================================================
# if os.getenv('COMPUTERNAME') == 'DESKTOP-DCG6K4F': #Jon's Work PC    
#     #set all desired recording parameters for this session        
#     #sessionID = 'test1_01_21' #create a session ID for output videos and CSV names
#     exposure = -6
#     resWidth = 960
#     resHeight = 720
#     framerate = 30
#     codec = 'DIVX' #other codecs to try include H264, DIVX
#     paramDict = {'exposure':exposure,"resWidth":resWidth,"resHeight":resHeight,'framerate':framerate,'codec':codec}
#     #for rotation inputs, specify a rotation for each camera as either: rotation0, rotation90, rotation180, or rotation270
#     #if rotating any camera, each camera needs to have a rotation input. i.e. [rotation90,rotation0,rotation0]
#     #if no rotations are needed, just have rotation_input = [] (an empty array)
#     rotation_input = []
# else:
#     exposure = -6
#     resWidth = 960
#     resHeight = 720
#     framerate = 30
#     codec = 'DIVX' #other codecs to try include H264, DIVX
#     paramDict = {'exposure':exposure,"resWidth":resWidth,"resHeight":resHeight,'framerate':framerate,'codec':codec}
#     rotation_input = []
#     
# 
# 
# if task == 'record':#don't change this boolean by accident pls
#     #recordPath = filepath/sessionID
#     #recordPath.mkdir(exist_ok='True')   
#     if rotation_input and not len(cam_inputs) == len(rotation_input):
#         raise ValueError('The number of camera inputs and rotation inputs does not match')
#     if not rotation_input:
#         rotation_input = [None]*len(cam_inputs)
#     is_empty = not any(sesh.baseFolder.iterdir())
#     if not is_empty:
#             raise RuntimeError(sesh.sessionID + ' folder already contains files. check session ID')
#     
#     if not cam_inputs:
#         raise ValueError('Camera input list (cam_inputs) is empty')
#     table = webcam.RunCams(cam_inputs,sesh.baseFolder,sesh.sessionID,paramDict,rotation_input) #press ESCAPE to end the recording
# =============================================================================
########################################################
##### Track Pixel Locations in synced videos
########################################################

########################################################
##### OpenPose Stuff!
########################################################
 
#sesh.RunOpenPose() #run videos through open pose and save json's out to a folder
#sesh.ParseOpenPose() #pump data from json's into dataframes for each camera view (sesh.openPoseDataFrames_nCams)  


########################################################
##### Calibrate capture volume (camera extrinsics)
########################################################

sesh.CalibrateCaptureVolume()



########################################################
##### Deeplabcut stuff!
########################################################
 
#sesh.ParseDLCdata() #just load in DLC data for now. Future iterations will use DLC methods for this
sesh.RunAndParseDeepLabCut(sesh.nSyncedVidFolder,sesh.dlcDataPath,sesh.DLCconfigPath)

########################################################
##### Reconstruct 3d stuff!!
########################################################

sesh.skel_fr_mar_dim = sesh.Reconstruct3D(sesh.openPoseData_nCams_nFrames_nImgPts_XYC) #oh boy, is it really happening?!

sesh.dlcPts_fr_mar_dim = sesh.Reconstruct3D(sesh.dlc_nCams_nFrames_nImgPts_XYC)

########################################################
##### Plot yrself a skeletron!!
########################################################
# vidType = 0 #No Cam Ims
# vidType = 1 #Only Cam1 im
# vidType = 2 #All Cam Ims
sesh.PlaySkeleton(1)