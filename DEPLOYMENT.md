# 🚢 Ship Carbon Footprint Deployment Guide

This guide describes how to deploy the **Ship CO₂ Emission Predictor** app in production. The app is built on Python (Flask), Scikit-Learn (predictive model), and Chart.js (interactive visualization).

---

## 🚀 Option 1: Deploy on Render (Recommended)

Render is the simplest platform for deploying this web application. We have pre-configured a `render.yaml` specification.

### Deployment Steps:
1. Push your repository to **GitHub** or **GitLab**.
2. Go to [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** and select **Blueprint**.
4. Connect your GitHub/GitLab repository.
5. Render will automatically read the `render.yaml` configuration and provision the service.

*Note: The app will run on Python 3.11.0 using `gunicorn app:app` and bind to the port defined by the environment variable `$PORT`.*

---

## 🐳 Option 2: Containerized Deployment via Docker

Using Docker ensures the application runs identically in any environment (AWS, GCP, Railway, Fly.io, etc.).

### 1. Build the Docker Image
Navigate to the root directory of the project and run:
```bash
docker build -t ship-carbon-predictor:latest .
```

### 2. Run the Container Locally
```bash
docker run -p 5000:5000 -e PORT=5000 ship-carbon-predictor:latest
```
Access the application at `http://localhost:5000`.

### 3. Deploy Docker on Render
1. Create a new **Web Service** on Render.
2. Connect your Git repository.
3. Select **Docker** as the Runtime (instead of Python).
4. Render will automatically detect the `Dockerfile` and build it.

---

## 💻 Option 3: Manual Deployment on Linux VPS

If you are running the app on a Virtual Private Server (Ubuntu/Debian):

### 1. Clone & Set Up Environment
```bash
git clone <your-repo-url> /var/www/ship-carbon-footprint
cd /var/www/ship-carbon-footprint

# Create Virtual Environment
python3 -m venv venv
source venv/bin/activate

# Install Dependencies
pip install -r requirements.txt
```

### 2. Configure Systemd Service
Create a systemd unit file to manage the Flask server process:
`sudo nano /etc/systemd/system/ship-carbon.service`

Add the following configuration:
```ini
[Unit]
Description=Gunicorn instance to serve Ship Carbon Predictor
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/var/www/ship-carbon-footprint
Environment="PATH=/var/www/ship-carbon-footprint/venv/bin"
ExecStart=/var/www/ship-carbon-footprint/venv/bin/gunicorn --workers 3 --bind 0.0.0.0:5000 app:app

[Install]
WantedBy=multi-user.target
```

### 3. Start the Service
```bash
sudo systemctl daemon-reload
sudo systemctl start ship-carbon
sudo systemctl enable ship-carbon
```

### 4. Set Up Nginx Reverse Proxy
Configure Nginx to route external requests on port 80 to the local port 5000:
`sudo nano /etc/nginx/sites-available/ship-carbon`

Add the configuration:
```nginx
server {
    listen 80;
    server_name your_domain_or_ip;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable the configuration and reload Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/ship-carbon /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```
