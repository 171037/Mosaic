from flask import Flask, render_template, Response
import cvlib as cv
import cv2

app = Flask(__name__)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1100)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

def generate_frames():
    while True:
        success, frame = cap.read()

        face, confidence = cv.detect_face(frame)
 
        for idx, f in enumerate(face):
        
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
    
            face_region = frame[startY:endY, startX:endX]
            
            M = face_region.shape[0]
            N = face_region.shape[1]

            if not face_region.size == 0:
                face_region = cv2.resize(face_region, None, fx=0.02, fy=0.02, interpolation=cv2.INTER_AREA)
                face_region = cv2.resize(face_region, (N, M), interpolation=cv2.INTER_AREA)
                frame[startY:endY, startX:endX] = face_region
        

        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
        
            
@app.route("/")
def home():
    return render_template('home.html')    

@app.route('/index.html')
def index():
    return render_template('index.html')




@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    app.run(debug=True)