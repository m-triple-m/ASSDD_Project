from flask import Flask, render_template, request, redirect, url_for, session, flash  
from flask_sqlalchemy import SQLAlchemy 
from werkzeug.security import generate_password_hash, check_password_hash 
from functools import wraps
from datetime import datetime
import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['RESULT_FOLDER'] = os.path.join('static', 'results')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

app.secret_key = os.getenv('SECRET_KEY', 'dev-secretkey-change-in-production')  # secret key for session management
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False # to suppress a warning from SQLAlchemy
db = SQLAlchemy(app) # initialize the database

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(200)) 
    # Relationship to cleanly pull all history instances belonging to this user
    histories = db.relationship('History', backref='user', lazy=True)

# History model
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    original_image = db.Column(db.String(255), nullable=False)
    mask_image = db.Column(db.String(255), nullable=False)
    overlay_image = db.Column(db.String(255), nullable=False)
    inference_time = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Database initialization with app context
with app.app_context(): 
    db.create_all()

# ---------------------------------------------------------
# 1. MODEL DEFINITION
# ---------------------------------------------------------
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
    )

class ResNetUNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.base_model = models.resnet18(pretrained=False)
        self.base_layers = list(self.base_model.children())
        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer0_1 = nn.Sequential(*self.base_layers[3:5])
        self.layer1, self.layer2, self.layer3 = self.base_layers[5], self.base_layers[6], self.base_layers[7]
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up3 = double_conv(512 + 256, 256)
        self.conv_up2 = double_conv(256 + 128, 128)
        self.conv_up1 = double_conv(128 + 64, 64)
        self.conv_up0 = double_conv(64 + 64, 32)
        self.conv_last = nn.Conv2d(32, n_class, 1)
        
    def forward(self, x):
        l0 = self.layer0(x)
        l0_1 = self.layer0_1(l0)
        l1 = self.layer1(l0_1)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        x = self.upsample(l3)
        x = torch.cat([x, l2], dim=1)
        x = self.conv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, l1], dim=1)
        x = self.conv_up2(x)
        x = self.upsample(x)
        x = F.interpolate(x, size=l0_1.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, l0_1], dim=1)
        x = self.conv_up1(x)
        x = self.upsample(x)
        x = F.interpolate(x, size=l0.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, l0], dim=1)
        x = self.conv_up0(x)
        out = self.upsample(self.conv_last(x))
        return out

# ---------------------------------------------------------
# 2. CLASS CONFIGURATION
# ---------------------------------------------------------
CLASS_LABELS = {
    0: "Unlabeled", 1: "Paved Area", 2: "Dirt", 3: "Grass", 
    4: "Gravel", 5: "Water", 6: "Rocks", 7: "Pool", 
    8: "Vegetation", 9: "Roof", 10: "Wall", 11: "Window", 
    12: "Door", 13: "Fence", 14: "Fence Pole", 15: "Person", 
    16: "Dog", 17: "Car", 18: "Bicycle", 19: "Tree", 
    20: "Bald Tree", 21: "Arid Vegetation", 22: "Obstacle"
}

COLOR_MAP = np.array([
    (0, 0, 0), (128, 128, 128), (150, 75, 0), (0, 154, 23), (192, 192, 192),
    (0, 0, 255), (105, 105, 105), (0, 255, 255), (0, 255, 0), (255, 0, 0),
    (165, 42, 42), (0, 191, 255), (255, 165, 0), (218, 165, 32), (184, 134, 11),
    (255, 192, 203), (255, 20, 147), (255, 255, 0), (127, 0, 255), (34, 139, 34),
    (210, 180, 140), (255, 215, 0), (128, 0, 0)
], dtype=np.uint8)

# ---------------------------------------------------------
# 3. LOAD MODEL
# ---------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetUNet(23).to(DEVICE)
if os.path.exists('resnetunet_aerial.pth'):
    sd = torch.load('resnetunet_aerial.pth', map_location=DEVICE)
    model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
    model.eval()

transform = transforms.Compose([
    transforms.Resize((512, 512)), # Best accuracy matches patch size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------------------
# 4. ROUTES
# ---------------------------------------------------------

# Public routes (no login required)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact')
def contact():
    return render_template('Contact.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    
    name = request.form['name']
    email = request.form['email']
    password = request.form['password']
    confirm_password = request.form['confirm_password']

    # validations
    if not name or len(name.strip()) < 2:
        flash('Name must be at least 2 characters long.', 'error')
        return redirect('/register')
    
    if not email or '@' not in email:
        flash('Please enter a valid email address.', 'error')
        return redirect('/register')
    
    # password must be at least 8 characters long and a combination of letters and numbers and special characters
    if len(password) < 8 or not any(char.isdigit() for char in password)\
          or not any(char.isalpha() for char in password) or not any(not char.isalnum()\
                                                                      for char in password):
        flash('Password must be at least 8 characters long and contain letters, numbers, and special characters.', 'error')
        return redirect('/register')
    
    if password != confirm_password:
        flash('Passwords do not match.', 'error')
        return redirect('/register')
    
    # check if user already exists
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        flash('Email already registered. Please log in.', 'error')
        return redirect('/login')
    
    # create new user
    hashed_password = generate_password_hash(password)
    new_user = User(
        name=name.strip(),
        email=email.strip(),
        password=hashed_password
    )
    try:
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect('/login')
    except Exception as e:
        db.session.rollback()
        flash('An error occurred during registration. Please try again.', 'error')
        return redirect(url_for('register'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    
    email = request.form['email']
    password = request.form['password']
    user = User.query.filter_by(email=email).first()

    if user and check_password_hash(user.password, password):
        session['user_id'] = user.id
        session['user_name'] = user.name
        flash('Login successful!', 'success')
        return redirect('/upload')
    else:
        flash('Invalid email or password.', 'error')
        return redirect('/login')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'success')
    return redirect('/')

# Protected routes (login required)
@app.route('/upload')
@app.route('/detector')
@login_required
def detector():
    return render_template('upload.html')

@app.route('/history')
@login_required
def history():
    user_history = History.query.filter_by(user_id=session['user_id']).order_by(History.timestamp.desc()).all()
    return render_template('history.html', histories=user_history)


@app.route('/predict', methods=['GET','POST'])
@login_required
def predict():
    file = request.files.get('file')
    if not file: 
        return redirect('/')
    
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    img_pil = Image.open(filepath).convert('RGB')
    orig_w, orig_h = img_pil.size
    input_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    
    start = time.time()
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    inf_time = round((time.time() - start) * 1000, 2)
    
    # Generate Legend & Stats
    unique_labels, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    stats = []
    for lbl, count in zip(unique_labels, counts):
        pct = round((count / total_pixels) * 100, 1)
        color = COLOR_MAP[lbl]
        hex_color = '#%02x%02x%02x' % (color[0], color[1], color[2])
        stats.append({'name': CLASS_LABELS.get(lbl, "Other"), 'pct': pct, 'color': hex_color})
    stats = sorted(stats, key=lambda x: x['pct'], reverse=True)

    # Prepare Visuals
    mask_rgb = COLOR_MAP[mask]
    mask_img = cv2.resize(mask_rgb, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    # Create Blended Overlay
    orig_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(orig_img, 0.6, cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR), 0.4, 0)
    
    mask_filename = 'mask_' + filename
    overlay_filename = 'overlay_' + filename

    cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], mask_filename), cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'], overlay_filename), overlay)
    
    # --- SAVE TO HISTORY DATABASE ---
    try:
        new_history = History(
            user_id=session['user_id'],
            original_image=filename,
            mask_image=mask_filename,
            overlay_image=overlay_filename,
            inference_time=inf_time
        )
        db.session.add(new_history)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Error saving history: {e}") # helpful for backend debugging
    # --------------------------------
    
    return render_template('predict.html', filename=filename, time=inf_time, stats=stats)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)