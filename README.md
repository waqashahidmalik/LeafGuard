
# **LeafGuard 🌿**  

LeafGuard is a Django-based API for **plant disease classification** using a machine learning model.  

## **🚀 Features**  
- Upload an image of a plant leaf  
- Get a **disease prediction** with a confidence score  

## **🛠️ Installation**  

### **Clone the Repository**  
```sh
git clone https://github.com/waqashahidmalik/LeafGuard.git
cd LeafGuard
```

### **Setup Virtual Environment (Recommended)**  
```sh
python -m venv venv  
source venv/bin/activate   # On macOS/Linux  
venv\Scripts\activate      # On Windows  
```

### **Install Dependencies**  
```sh
pip install -r requirements.txt
```

### **Run Migrations**  
```sh
python manage.py migrate
```

### **Start the Server**  
```sh
python manage.py runserver
```

## **📂 Project Structure**  
```
LeafGuard/
│── api/                # Django app for API
│── LeafGuard/          # Django project settings
│── manage.py           # Django management script
│── requirements.txt    # Dependencies
│── README.md           # Documentation
└── .gitignore          # Ignored files
```

## **🚀 API Usage**  
Send a `POST` request with an image:  
```sh
curl -X POST -F "image=@path/to/image.jpg" http://127.0.0.1:8000/api/predict/
```

## **📌 Notes**  
- **Model files are not included**. Place them in the appropriate directory.  

