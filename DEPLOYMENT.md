# Backend Deployment Guide

This guide covers deploying the Synthetic Workforce Simulator backend to various platforms.

## Prerequisites

- Docker and Docker Compose installed
- PostgreSQL database (or use managed service)
- Groq API keys (at least one, multiple recommended)
- Domain name (optional, for production)

---

## Option 1: Docker Compose (Recommended for Development/Staging)

### Step 1: Set Up Environment Variables

Create a `.env` file in the `synthetic-workforce-simulator/` directory:

```bash
# Database
DB_DATABASE_URL=postgresql+asyncpg://postgres:postgres@postgres:5432/workforce_simulator

# Groq API Keys (use multiple for load distribution)
GROQ_API_KEY1=your_groq_key_1_here
GROQ_API_KEY2=your_groq_key_2_here
GROQ_API_KEY3=your_groq_key_3_here

# Optional: Single key fallback
# GROQ_API_KEY=your_single_key_here

# Groq Settings
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_MAX_TOKENS=2048
GROQ_TEMPERATURE=0.7
GROQ_TIMEOUT=30

# Application
APP_NAME=Synthetic Workforce Simulator
ENVIRONMENT=production
DEBUG=false
HOST=0.0.0.0
PORT=8000

# Security
SECRET_KEY=your-secret-key-change-in-production-min-32-chars
CORS_ORIGINS=https://your-frontend-domain.com,http://localhost:3000

# Redis (optional)
REDIS_URL=redis://redis:6379/0

# Logging
LOG_LEVEL=INFO
JSON_LOGS=true
```

### Step 2: Build and Start Services

From the project root directory:

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f workforce-simulator-app

# Check status
docker-compose ps
```

### Step 3: Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# API docs
open http://localhost:8000/docs
```

### Step 4: Stop Services

```bash
docker-compose down

# Remove volumes (clears database)
docker-compose down -v
```

---

## Option 2: AWS EC2 Deployment

### Step 1: Launch EC2 Instance

1. Go to AWS Console → EC2 → Launch Instance
2. Choose Ubuntu 22.04 LTS
3. Instance type: t3.medium or larger (2+ vCPU, 4+ GB RAM)
4. Configure security group:
   - Port 22 (SSH)
   - Port 80 (HTTP)
   - Port 443 (HTTPS)
   - Port 8000 (API, optional for testing)

### Step 2: Connect to Instance

```bash
ssh -i your-key.pem ubuntu@your-ec2-ip
```

### Step 3: Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Logout and login again for docker group to take effect
exit
```

### Step 4: Clone Repository

```bash
# Install git
sudo apt install git -y

# Clone your repository
git clone https://github.com/yourusername/synthwork.git
cd synthwork
```

### Step 5: Set Up Environment

```bash
cd synthetic-workforce-simulator

# Create .env file
nano .env
# Paste your environment variables (see Option 1, Step 1)
```

### Step 6: Set Up PostgreSQL (if not using managed service)

```bash
# In docker-compose.yml, ensure postgres service is configured
# Or use AWS RDS for managed PostgreSQL
```

### Step 7: Deploy

```bash
# From project root
docker-compose up -d

# Check logs
docker-compose logs -f workforce-simulator-app
```

### Step 8: Set Up Nginx Reverse Proxy (Optional but Recommended)

```bash
# Install Nginx
sudo apt install nginx -y

# Create Nginx config
sudo nano /etc/nginx/sites-available/synthwork

# Add this configuration:
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/synthwork /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Step 9: Set Up SSL with Let's Encrypt

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx -y

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal is set up automatically
```

---

## Option 3: Google Cloud Platform (GCP) Deployment

### Step 1: Create Compute Engine Instance

1. Go to GCP Console → Compute Engine → VM Instances
2. Create Instance:
   - Name: synthwork-backend
   - Machine type: e2-medium or larger
   - Boot disk: Ubuntu 22.04 LTS
   - Firewall: Allow HTTP, HTTPS traffic

### Step 2: Follow AWS Steps 2-7

Same as AWS deployment steps 2-7.

### Step 3: Set Up Cloud Load Balancer (Optional)

1. Go to Network Services → Load Balancing
2. Create HTTP(S) Load Balancer
3. Backend: Your VM instance
4. Frontend: Configure IP and SSL certificate

---

## Option 4: Railway Deployment

### Step 1: Create Railway Account

1. Go to https://railway.app
2. Sign up with GitHub

### Step 2: Create New Project

1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Connect your repository

### Step 3: Configure Services

1. Add PostgreSQL service (Railway provides managed PostgreSQL)
2. Add your backend service

### Step 4: Set Environment Variables

In Railway dashboard, add all environment variables from `.env` file.

### Step 5: Deploy

Railway automatically deploys on git push to main branch.

---

## Option 5: Render Deployment

### Step 1: Create Render Account

1. Go to https://render.com
2. Sign up with GitHub

### Step 2: Create Web Service

1. Click "New" → "Web Service"
2. Connect your GitHub repository
3. Select the `synthetic-workforce-simulator` directory

### Step 3: Configure Build

- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Step 4: Add PostgreSQL Database

1. Click "New" → "PostgreSQL"
2. Copy the database URL

### Step 5: Set Environment Variables

Add all environment variables in the Render dashboard.

### Step 6: Deploy

Render automatically deploys on git push.

---

## Option 6: DigitalOcean App Platform

### Step 1: Create DigitalOcean Account

1. Go to https://www.digitalocean.com
2. Sign up

### Step 2: Create App

1. Go to App Platform
2. Click "Create App"
3. Connect GitHub repository

### Step 4: Configure App

- **Type**: Web Service
- **Source**: `synthetic-workforce-simulator/`
- **Build Command**: `pip install -r requirements.txt`
- **Run Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Step 5: Add Database

1. Add PostgreSQL database component
2. Copy connection string

### Step 6: Set Environment Variables

Add all environment variables.

### Step 7: Deploy

DigitalOcean automatically deploys on git push.

---

## Post-Deployment Checklist

- [ ] Health check endpoint returns 200
- [ ] API docs accessible at `/docs`
- [ ] Database connection working
- [ ] Groq API keys configured and working
- [ ] CORS configured for frontend domain
- [ ] SSL certificate installed (production)
- [ ] Logs are being generated
- [ ] Environment variables are set correctly
- [ ] Rate limiting is working
- [ ] Multiple API keys are rotating properly

---

## Monitoring and Maintenance

### View Logs

```bash
# Docker Compose
docker-compose logs -f workforce-simulator-app

# Docker
docker logs -f <container-id>
```

### Update Deployment

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose up -d --build
```

### Backup Database

```bash
# Export database
docker-compose exec postgres pg_dump -U postgres workforce_simulator > backup.sql

# Restore database
docker-compose exec -T postgres psql -U postgres workforce_simulator < backup.sql
```

### Scale Services

```bash
# Scale API service
docker-compose up -d --scale workforce-simulator-app=3
```

---

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
sudo lsof -i :8000

# Kill process
sudo kill -9 <PID>
```

### Database Connection Issues

```bash
# Check database is running
docker-compose ps postgres

# Check database logs
docker-compose logs postgres

# Test connection
docker-compose exec postgres psql -U postgres -d workforce_simulator
```

### API Key Issues

```bash
# Check environment variables
docker-compose exec workforce-simulator-app env | grep GROQ

# Test API key
curl -H "Authorization: Bearer YOUR_KEY" https://api.groq.com/openai/v1/models
```

### Rate Limit Issues

- Add more Groq API keys
- Check key rotation is working
- Monitor error logs for rate limit messages

---

## Security Best Practices

1. **Never commit `.env` files** - Use environment variables in deployment platform
2. **Use strong SECRET_KEY** - Generate with: `python -c "import secrets; print(secrets.token_urlsafe(32))"`
3. **Enable HTTPS** - Always use SSL in production
4. **Restrict CORS** - Only allow your frontend domain
5. **Use managed databases** - AWS RDS, GCP Cloud SQL, etc.
6. **Regular updates** - Keep dependencies updated
7. **Monitor logs** - Set up log aggregation (CloudWatch, Datadog, etc.)

---

## Cost Optimization

1. **Use spot instances** for development/staging
2. **Right-size instances** - Monitor CPU/memory usage
3. **Use managed databases** - Often cheaper than self-hosted
4. **Optimize API calls** - Cache responses when possible
5. **Use CDN** - For static assets (if any)

---

## Support

For issues or questions:
- Check logs: `docker-compose logs -f`
- Review API docs: `http://your-domain/docs`
- Check health: `http://your-domain/health`

