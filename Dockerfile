FROM python:3

# Add everything from the folder to a folder in the docker image root
ADD . ./face_blurry_app 
WORKDIR ./face_blurry_app

# Install all requirements
RUN pip install -r requirements.txt

# Running the app with default values
CMD [ "python", "src/blur_face_in_img.py"]
