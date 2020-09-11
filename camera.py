import cv2
import boto3
import datetime
import requests
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6

count=0

class VideoCamera(object):    
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        #count=0
        global count
        success, image = self.video.read()
        is_success, im_buf_arr = cv2.imencode(".jpg", image)
        image1 = im_buf_arr.tobytes()
        client=boto3.client('rekognition',
                        aws_access_key_id="ASIA3JZX6DJK2IOZBPEV",
                        aws_secret_access_key="GQ4AuQs80d8r+gLfQCadeLY/vmll0SLFPQMF/x9P",
                        aws_session_token="FwoGZXIvYXdzEHoaDMK6+Vqt+bc4zxdiSyLKAe9iC6fIvoALw6dZuXTSz5Vb0GfE43zPfJTLsmHOA+pDUpGwlCEBfT6xXrgPq5XiGabwP/5ZFbp517LpM08a3f76c356zrXXYSVPazZogFUMc/qMDkEWly/SW66SeT9cgRirmZAj49GMGUBAFovwnWAUOmWEMJVOT+R7BCcRDs7qzlV8mrmhichmPsmSWqOcZsJY+2b99WyupvX8XorhsQepP0eQK0VkZVxU0FN1iFgijdC1FgZ51y0fKVfkXFbONQ2CXdn0EnAYOcAoqu3s+gUyLRhXqAddoXMzN2yXr8kKsDW9H2XiMzfy4lVX669OchDI696RMMVo3K66fvIdiA==",
                        region_name='us-east-1')
                               
        response = client.detect_custom_labels(
        ProjectVersionArn='arn:aws:rekognition:us-east-1:776969525845:project/Mask-Detection2/version/Mask-Detection2.2020-09-07T23.02.02/1599499928143',Image={
            'Bytes':image1})
        print(response['CustomLabels'])
        
        if not len(response['CustomLabels']):
            count=count+1
            date = str(datetime.datetime.now()).split(" ")[0]
            #print(date)
            url = " https://81ryisfwlc.execute-api.us-east-1.amazonaws.com/apiForMaskCount?date="+date+"&count="+str(count)
            resp = requests.get(url)
            f = open("countfile.txt", "w")
            f.write(str(count))
            f.close()
            #print(count)

        image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
        	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        	break
        ret, jpeg = cv2.imencode('.jpg', image)
        #cv2.putText(image, text = str(count), org=(10,40), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(1,0,0))
        cv2.imshow('image',image)
        return jpeg.tobytes()
