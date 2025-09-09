# Face Recognition App

A Python-based **facial recognition system** built using [OpenCV](https://opencv.org/) and [face_recognition](https://github.com/ageitgey/face_recognition).  
This application can detect and recognize faces from a live webcam feed and display stored **metadata** (such as name, age, and height).  

---

## Features
- Real-time face recognition using your computer's webcam.  
- Matches detected faces against a folder of **known faces**.  
- Displays **user metadata** (name, age, height) from `user_data.json`.  
- Simple setup — just add images of known users and their details.  
- Extensible: can integrate with smart home devices (e.g., greet a person, control IoT devices).  

---

## Project Structure
```bash
face-recognition-app/
├── detect.py # Main application script
├── known_faces/ # Folder containing reference images (e.g., person1.jpg)
├── user_data.json # Metadata for known users
├── requirements.txt # Python dependencies
└── README.md # Project documentation
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/sbaru123/face-recognition-app.git
cd face-recognition-app
```

### 2. Create & Activate Virtual Environment
```bash
conda create -n faceapp python=3.11
conda activate faceapp
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Add Known Faces
Place .jpg images inside the known_faces/ folder.
Example:
```bash
known_faces/john.jpg
known_faces/sarah.jpg
```

### 5. Update user_data.json

```bash
{
  "john": { "name": "John Doe", "age": 25, "height": "6ft" },
  "sarah": { "name": "Sarah Lee", "age": 22, "height": "5'6" }
}
```
### 6. Run the App
```bash
python detect.py
```
Press Q to quit the camera feed. 

### Contributing
Pull requests are welcome! Please fork this repo and submit a PR.

### License
This project is licensed under the MIT License.









