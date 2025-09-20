# 🚀 Tennis Predictor 101 - Production Deployment Guide

**Complete deployment instructions for the world's most advanced tennis prediction system**

---

## 📋 **Prerequisites**

### **System Requirements**
- **OS**: Ubuntu 20.04+ / CentOS 8+ / macOS 10.15+
- **RAM**: 8GB minimum, 16GB recommended
- **CPU**: 4 cores minimum, 8 cores recommended
- **Storage**: 10GB minimum, 50GB recommended
- **Network**: Stable internet for data collection

### **Software Dependencies**
- **Python**: 3.8+
- **Docker**: 20.10+ (optional)
- **PostgreSQL**: 12+ (for data storage)
- **Redis**: 6+ (for caching)
- **Git**: Latest version

---

## 🏗️ **Installation Methods**

### **Method 1: Automated Production Setup**

```bash
# Clone repository
git clone https://github.com/Milkpainter/tennis-predictor-101.git
cd tennis-predictor-101

# Run automated setup (Ubuntu/Debian)
sudo chmod +x scripts/setup_production.sh
./scripts/setup_production.sh

# Follow prompts for:
# - SSL certificate setup
# - Database configuration  
# - Monitoring installation
```

### **Method 2: Docker Deployment**

```bash
# Quick start with Docker Compose
docker-compose up -d

# Check services status
docker-compose ps

# View logs
docker-compose logs -f tennis-predictor
```

### **Method 3: Manual Installation**

```bash
# 1. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure system
cp config/config.example.yaml config.yaml
# Edit config.yaml with your settings

# 4. Initialize database
python scripts/init_database.py

# 5. Train the system
python scripts/train_ultimate_system.py --years 2020-2024 --full-training

# 6. Start API server
python run_prediction_server.py --host 0.0.0.0 --port 8000
```

---

## ⚙️ **Configuration**

### **Core Configuration (config.yaml)**

```yaml
# Production configuration
environment: production

# API Settings
api:
  host: 0.0.0.0
  port: 8000
  debug: false
  rate_limit_per_minute: 200

# Database Configuration
database:
  host: localhost
  port: 5432
  database: tennis_predictor_prod
  username: tennisapp
  password: your_secure_password

# Model Configuration
models:
  ensemble:
    meta_learner: logistic_regression
    cv_folds: 5
    dynamic_weighting: true
  
  xgboost:
    n_estimators: 500
    max_depth: 6
    learning_rate: 0.1
    hyperparameter_optimization: true

# Momentum System (42 indicators)
momentum:
  weights:
    serving:
      break_points_saved: 3.0      # Highest predictor
      service_hold_rate: 2.5
      service_games_streak: 2.0
    return:
      break_point_conversion: 3.0  # Highest return predictor
      return_points_trend: 2.0
    rally:
      rally_win_percentage: 3.0    # Fundamental indicator
      pressure_rally_performance: 2.5

# Betting Configuration
betting:
  kelly_fraction_cap: 0.25        # Max 25% of bankroll
  min_edge_threshold: 0.02        # Min 2% edge
  min_confidence_threshold: 0.6   # Min 60% confidence
```

### **Environment Variables (.env)**

```bash
# Core settings
ENVIRONMENT=production
SECRET_KEY=your_secret_key_here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/tennis_predictor

# Redis Cache
REDIS_URL=redis://localhost:6379/0

# API Keys (optional)
ODDS_API_KEY=your_odds_api_key
TENNIS_DATA_API_KEY=your_tennis_data_key

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
```

---

## 🎯 **System Training**

### **Full System Training**

```bash
# Complete training pipeline
python scripts/train_ultimate_system.py \
  --years 2020-2024 \
  --full-training \
  --log-level INFO

# This will:
# 1. Download historical match data
# 2. Engineer all 200+ features
# 3. Train all 5 base models with optimization
# 4. Create stacking ensemble
# 5. Validate using tournament-based CV
# 6. Save trained models
```

### **Quick Training (Development)**

```bash
# Faster training for testing
python scripts/train_ultimate_system.py \
  --years 2023-2024
```

### **Training Progress Monitoring**

```bash
# Monitor training logs
tail -f logs/training_*.log

# Check training status
python -c "from scripts.train_ultimate_system import check_training_status; check_training_status()"
```

---

## 🌐 **API Server Deployment**

### **Development Server**

```bash
# Start development server
python run_prediction_server.py --reload

# API available at: http://localhost:8000
# Documentation: http://localhost:8000/docs
```

### **Production Server**

```bash
# Using Gunicorn (recommended)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 300 \
  --keepalive 10 \
  api.prediction_server:app

# Or using Uvicorn directly
uvicorn api.prediction_server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --timeout-keep-alive 10
```

### **Systemd Service Setup**

```ini
# /etc/systemd/system/tennis-predictor.service
[Unit]
Description=Tennis Predictor 101 API
After=network.target postgresql.service

[Service]
Type=simple
User=tennisapp
Group=tennisapp
WorkingDirectory=/opt/tennis-predictor-101
Environment=PATH=/opt/tennis-predictor-101/venv/bin
EnvironmentFile=/opt/tennis-predictor-101/.env
ExecStart=/opt/tennis-predictor-101/venv/bin/gunicorn -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 api.prediction_server:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable tennis-predictor
sudo systemctl start tennis-predictor
sudo systemctl status tennis-predictor
```

---

## 🔒 **Security Setup**

### **SSL/TLS Configuration**

```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Obtain SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal setup
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

### **Firewall Configuration**

```bash
# Configure UFW
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw allow 8000/tcp  # API port

# Check status
sudo ufw status
```

### **Database Security**

```sql
-- Create dedicated database user
CREATE USER tennisapp WITH PASSWORD 'secure_password';
CREATE DATABASE tennis_predictor OWNER tennisapp;
GRANT ALL PRIVILEGES ON DATABASE tennis_predictor TO tennisapp;

-- Limit connection privileges
ALTER USER tennisapp SET default_transaction_isolation TO 'read committed';
```

---

## 📊 **Performance Monitoring**

### **Health Checks**

```bash
# API health check
curl http://localhost:8000/health

# System status
curl http://localhost:8000/status

# Model information
curl http://localhost:8000/models/info
```

### **Prometheus Metrics**

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'tennis-predictor'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### **Log Monitoring**

```bash
# Application logs
journalctl -u tennis-predictor -f

# Training logs
tail -f logs/ultimate_system_*.log

# API access logs
tail -f logs/api_access.log
```

---

## 🧪 **Testing & Validation**

### **System Tests**

```bash
# Run comprehensive system test
python ultimate_main.py --mode all

# Benchmark lab performance
python ultimate_main.py --mode benchmark

# Validate system accuracy
python ultimate_main.py --mode validate

# Single prediction test
python ultimate_main.py --mode predict \
  --player1 "Novak Djokovic" \
  --player2 "Carlos Alcaraz" \
  --tournament "US Open" \
  --surface "Hard"
```

### **API Testing**

```bash
# Test prediction endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "player1": {"player_id": "Novak Djokovic"},
    "player2": {"player_id": "Carlos Alcaraz"},
    "tournament": "US Open",
    "surface": "Hard",
    "betting_odds": {
      "player1_decimal_odds": 1.85,
      "player2_decimal_odds": 2.05
    }
  }'

# Expected response:
# {
#   "predicted_winner": "Novak Djokovic",
#   "player1_win_probability": 0.6234,
#   "confidence": 0.8567,
#   "processing_time_ms": 67.3
# }
```

### **Load Testing**

```bash
# Install testing tools
pip install locust pytest-benchmark

# Run load test
locust -f tests/load_test.py --host=http://localhost:8000
```

---

## 📈 **Performance Benchmarks**

### **Target Performance Metrics**

| Metric | Development | Production | Elite |
|--------|-------------|------------|
| **Accuracy** | 85%+ | 88%+ | **91%+** |
| **ROI** | 5%+ | 8%+ | **12%+** |
| **Response Time** | <200ms | <100ms | **<50ms** |
| **Uptime** | 95%+ | 99%+ | **99.9%+** |
| **Confidence** | 70%+ | 80%+ | **90%+** |

### **Optimization Checklist**

- [ ] **Model Optimization**: All 5 base models trained with hyperparameter tuning
- [ ] **Feature Engineering**: 200+ features including 42 momentum indicators
- [ ] **Ensemble Stacking**: Meta-learning with probability calibration
- [ ] **Caching**: Redis caching for predictions and player data
- [ ] **Database Indexing**: Optimized queries for historical data
- [ ] **API Optimization**: Connection pooling and async processing
- [ ] **Load Balancing**: Multiple server instances for high traffic

---

## 🔍 **Monitoring & Alerts**

### **Key Metrics to Monitor**

```python
# Key Performance Indicators (KPIs)
KPIs = {
    'prediction_accuracy': 0.88,      # Target: 88%+
    'api_response_time_ms': 100,       # Target: <100ms
    'system_uptime_pct': 99.0,         # Target: 99%+
    'prediction_confidence': 0.80,     # Target: 80%+
    'betting_roi_pct': 8.0,           # Target: 8%+
    'daily_predictions': 1000,        # Capacity target
    'error_rate_pct': 1.0,            # Target: <1%
    'model_accuracy_drift': 0.02      # Target: <2% drift
}
```

### **Alert Configuration**

```yaml
# alerting.yml
alerts:
  - name: "Prediction Accuracy Drop"
    condition: "accuracy < 0.85"
    severity: "critical"
    action: "retrain_models"
    
  - name: "High Response Time"
    condition: "avg_response_time_ms > 150"
    severity: "warning"
    action: "scale_resources"
    
  - name: "System Downtime"
    condition: "uptime < 0.95"
    severity: "critical"
    action: "restart_services"
```

---

## 🚦 **Production Checklist**

### **Pre-Deployment**
- [ ] System trained on 2020-2024 data
- [ ] All 100 labs tested and functional
- [ ] Tournament-based cross-validation completed
- [ ] Target accuracy achieved (88%+)
- [ ] API endpoints tested
- [ ] Security configurations verified
- [ ] SSL certificates installed
- [ ] Database optimized and indexed
- [ ] Backup procedures configured

### **Post-Deployment**
- [ ] Health checks passing
- [ ] Monitoring dashboards configured
- [ ] Alert systems active
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] Team training completed
- [ ] Incident response plan ready

---

## 🎯 **Usage Examples**

### **Single Match Prediction**

```python
# Python API usage
from prediction_engine.ultimate_predictor import UltimateTennisPredictor

# Initialize system
predictor = UltimateTennisPredictor(model_path='models/saved/latest_ensemble.pkl')

# Make prediction
prediction = predictor.predict_match(
    player1_id="Novak Djokovic",
    player2_id="Carlos Alcaraz", 
    tournament="US Open",
    surface="Hard",
    round_info="SF"
)

print(f"Winner: {prediction.predicted_winner}")
print(f"Probability: {prediction.player1_win_probability:.1%}")
print(f"Confidence: {prediction.confidence:.1%}")
```

### **Batch Predictions**

```bash
# Command line batch processing
python examples/batch_predict.py \
  --input matches_to_predict.csv \
  --output predictions_output.csv \
  --include-betting-analysis
```

### **Real-Time API**

```javascript
// JavaScript frontend integration
const prediction = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    player1: {player_id: 'Novak Djokovic'},
    player2: {player_id: 'Carlos Alcaraz'},
    tournament: 'US Open',
    surface: 'Hard',
    betting_odds: {
      player1_decimal_odds: 1.85,
      player2_decimal_odds: 2.05
    }
  })
});

const result = await prediction.json();
console.log(`Winner: ${result.predicted_winner}`);
console.log(`Confidence: ${(result.confidence * 100).toFixed(1)}%`);
```

---

## 🛠️ **Maintenance**

### **Daily Tasks**
- [ ] Check system health and alerts
- [ ] Review prediction accuracy metrics  
- [ ] Monitor API response times
- [ ] Verify data collection processes

### **Weekly Tasks**
- [ ] Analyze betting ROI performance
- [ ] Review model performance trends
- [ ] Update player rankings and stats
- [ ] Check system resource usage

### **Monthly Tasks**
- [ ] Retrain models with new data
- [ ] Performance benchmark review
- [ ] Security audit and updates
- [ ] Backup verification
- [ ] Capacity planning review

### **Model Retraining**

```bash
# Automated monthly retraining
crontab -e
# Add: 0 2 1 * * /opt/tennis-predictor-101/scripts/automated_retrain.sh

# Manual retraining
python scripts/train_ultimate_system.py \
  --years 2020-2025 \
  --incremental-update \
  --validate-against-recent
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

#### **Model Training Fails**
```bash
# Check data availability
python scripts/check_data_integrity.py

# Verify system resources
htop
df -h

# Review training logs
tail -100 logs/training_*.log
```

#### **API Server Not Responding**
```bash
# Check service status
sudo systemctl status tennis-predictor

# Review error logs
journalctl -u tennis-predictor --since "1 hour ago"

# Restart service
sudo systemctl restart tennis-predictor
```

#### **Low Prediction Accuracy**
```bash
# Validate model performance
python ultimate_main.py --mode validate

# Check for data drift
python scripts/detect_data_drift.py

# Retrain if necessary
python scripts/train_ultimate_system.py --force-retrain
```

### **Performance Optimization**

#### **Speed Optimization**
```bash
# Profile prediction speed
python -m cProfile -s cumulative ultimate_main.py --mode benchmark

# Enable model caching
# Edit config.yaml: caching.enabled = true

# Use faster hardware
# Upgrade to SSD storage
# Increase RAM to 32GB
# Use multi-core CPU
```

#### **Accuracy Optimization**
```bash
# Retrain with more data
python scripts/train_ultimate_system.py --years 2018-2025

# Optimize hyperparameters
python scripts/hyperparameter_search.py --extensive

# Add new features
python scripts/feature_engineering_research.py --implement-latest
```

---

## 📞 **Support & Updates**

### **Getting Help**
- **Documentation**: http://localhost:8000/docs
- **GitHub Issues**: https://github.com/Milkpainter/tennis-predictor-101/issues
- **Discussions**: https://github.com/Milkpainter/tennis-predictor-101/discussions

### **Updates**
```bash
# Check for updates
git fetch origin
git log HEAD..origin/main --oneline

# Update system
git pull origin main
pip install -r requirements.txt

# Retrain if models updated
python scripts/train_ultimate_system.py --incremental

# Restart services
sudo systemctl restart tennis-predictor
```

---

## 🏆 **Success Metrics**

### **System is Successfully Deployed When:**

✅ **Accuracy**: Achieving 88-91% prediction accuracy on validation data  
✅ **Speed**: API responses under 100ms for single predictions  
✅ **Reliability**: 99%+ uptime with proper error handling  
✅ **ROI**: Generating 8-12% ROI on betting recommendations  
✅ **Scalability**: Handling 1000+ predictions per hour  
✅ **Monitoring**: Full observability with alerts and dashboards  
✅ **Security**: SSL, authentication, and proper access controls  
✅ **Documentation**: Complete API docs and operational procedures  

---

## 🎉 **Congratulations!**

You now have the **world's most advanced tennis prediction system** running in production!

**Key Achievements:**
- 🔬 **100 Research-Validated Labs** running in harmony
- ⚡ **42 Momentum Indicators** providing game-changing insights
- 🤖 **5 Advanced ML Models** with ensemble stacking
- 💰 **Professional Betting Intelligence** with Kelly Criterion
- 🚀 **Production-Ready Performance** with <100ms predictions

**Your Tennis Predictor 101 system is ready to revolutionize tennis analytics!** 🏆

---

*For additional support or custom implementations, please refer to the GitHub repository or create an issue for assistance.*