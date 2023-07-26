import cv2
import numpy as np
from keras.models import load_model
from twilio.rest import Client

# Load the pre-trained Keras model for eye classification (open/closed)
model = load_model('bestModel.h5')  # Update with your model's path

# Load the pre-trained Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Access the webcam
cap = cv2.VideoCapture(0)

# Initialize variables for drowsiness detection
closed_eyes_counter = 0
drowsiness_threshold = 30  # Number of consecutive closed eye frames before alerting
alert = False

# Twilio account settings
twilio_account_sid = 'YOUR_TWILIO_ACCOUNT_SID'
twilio_auth_token = 'YOUR_TWILIO_AUTH_TOKEN'
twilio_phone_number = 'YOUR_TWILIO_PHONE_NUMBER'  # The Twilio phone number you obtained earlier
phone_number_to_call = 'CUSTOMER_PHONE_NUMBER'   # Replace wit

twilio_client = Client(twilio_account_sid, twilio_auth_token)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Get the region of interest (ROI) within the face and detect eyes
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # Preprocess the eye image for classification
            eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
            eye_roi = cv2.resize(eye_roi, (64, 64))  # Resize to (64, 64)
            eye_roi = eye_roi.astype("float") / 255.0
            eye_roi = np.expand_dims(eye_roi, axis=-1)  # Add a single channel (gray scale)
            eye_roi = np.expand_dims(eye_roi, axis=0)  # Add a batch dimension

            # Use the model to predict if the eye is open or closed
            prediction = model.predict(eye_roi)

            # Assuming the prediction shape is (batch_size, 1), update the label logic
            if prediction[0, 0] > 0.5:  # Assuming 0.5 is the threshold for binary classification
                label = "Open"
            else:
                label = "Closed"

            # Draw a rectangle around the eye and display the label
            color = (0, 255, 0) if label == "Open" else (0, 0, 255)
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), color, 1)
            cv2.putText(frame, label, (x + ex, y + ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Check for closed eyes and count consecutive frames
            if label == "Closed":
                closed_eyes_counter += 1
            else:
                closed_eyes_counter = 0

            # Alert the driver if eyes are closed for the specified threshold
            if closed_eyes_counter >= drowsiness_threshold and not alert:
                # Make a call using Twilio
                twilio_client.calls.create(
                    twiml='<Response><Say>Alert! Drowsiness Detected. Please wake up.</Say></Response>',
                    to=phone_number_to_call,
                    from_=twilio_phone_number
                )

                alert = True

    # Show the processed frame
    cv2.imshow("Drowsiness Detection", frame)

    # Break the loop if alert is triggered
    if alert:
        break

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
